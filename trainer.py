from torch.cuda.amp import autocast
from transformers import Seq2SeqTrainer, Adafactor, AdamW
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from packaging import version
from torch import nn
from torch.utils.data.dataset import Dataset
from transformers.deepspeed import is_deepspeed_zero3_enabled
from transformers.dependency_versions_check import dep_version_check
from transformers.file_utils import is_sagemaker_mp_enabled
from transformers.integrations import is_fairscale_available

from transformers.trainer import Trainer
from transformers.trainer_pt_utils import get_parameter_names
from transformers.trainer_utils import PredictionOutput, ShardedDDPOption
from transformers.utils import logging


if is_fairscale_available():
    dep_version_check("fairscale")
    from transformers.fairscale.optim import OSS

if is_sagemaker_mp_enabled():
    import smdistributed.modelparallel.torch as smp



class Seq2SeqEncoderParallelTrainer(Trainer):
    def evaluate(
            self,
            eval_dataset: Optional[Dataset] = None,
            ignore_keys: Optional[List[str]] = None,
            metric_key_prefix: str = "eval",
            max_length: Optional[int] = None,
            num_beams: Optional[int] = None,
    ) -> Dict[str, float]:
        """
        Run evaluation and returns metrics.

        The calling script will be responsible for providing a method to compute metrics, as they are task-dependent
        (pass it to the init :obj:`compute_metrics` argument).

        You can also subclass and override this method to inject custom behavior.

        Args:
            eval_dataset (:obj:`Dataset`, `optional`):
                Pass a dataset if you wish to override :obj:`self.eval_dataset`. If it is an :obj:`datasets.Dataset`,
                columns not accepted by the ``model.forward()`` method are automatically removed. It must implement the
                :obj:`__len__` method.
            ignore_keys (:obj:`List[str]`, `optional`):
                A list of keys in the output of your model (if it is a dictionary) that should be ignored when
                gathering predictions.
            metric_key_prefix (:obj:`str`, `optional`, defaults to :obj:`"eval"`):
                An optional prefix to be used as the metrics key prefix. For example the metrics "bleu" will be named
                "eval_bleu" if the prefix is ``"eval"`` (default)
            max_length (:obj:`int`, `optional`):
                The maximum target length to use when predicting with the generate method.
            num_beams (:obj:`int`, `optional`):
                Number of beams for beam search that will be used when predicting with the generate method. 1 means no
                beam search.

        Returns:
            A dictionary containing the evaluation loss and the potential metrics computed from the predictions. The
            dictionary also contains the epoch number which comes from the training state.
        """
        self._max_length = max_length
        self._num_beams = num_beams
        return super().evaluate(eval_dataset, ignore_keys=ignore_keys, metric_key_prefix=metric_key_prefix)

    def predict(
            self,
            test_dataset: Dataset,
            ignore_keys: Optional[List[str]] = None,
            metric_key_prefix: str = "eval",
            max_length: Optional[int] = None,
            num_beams: Optional[int] = None,
    ) -> PredictionOutput:
        """
        Run prediction and returns predictions and potential metrics.

        Depending on the dataset and your use case, your test dataset may contain labels. In that case, this method
        will also return metrics, like in :obj:`evaluate()`.

        Args:
            test_dataset (:obj:`Dataset`):
                Dataset to run the predictions on. If it is an :obj:`datasets.Dataset`, columns not accepted by the
                ``model.forward()`` method are automatically removed. Has to implement the method :obj:`__len__`
            ignore_keys (:obj:`List[str]`, `optional`):
                A list of keys in the output of your model (if it is a dictionary) that should be ignored when
                gathering predictions.
            metric_key_prefix (:obj:`str`, `optional`, defaults to :obj:`"eval"`):
                An optional prefix to be used as the metrics key prefix. For example the metrics "bleu" will be named
                "eval_bleu" if the prefix is ``"eval"`` (default)
            max_length (:obj:`int`, `optional`):
                The maximum target length to use when predicting with the generate method.
            num_beams (:obj:`int`, `optional`):
                Number of beams for beam search that will be used when predicting with the generate method. 1 means no
                beam search.

        .. note::

            If your predictions or labels have different sequence lengths (for instance because you're doing dynamic
            padding in a token classification task) the predictions will be padded (on the right) to allow for
            concatenation into one array. The padding index is -100.

        Returns: `NamedTuple` A namedtuple with the following keys:

            - predictions (:obj:`np.ndarray`): The predictions on :obj:`test_dataset`.
            - label_ids (:obj:`np.ndarray`, `optional`): The labels (if the dataset contained some).
            - metrics (:obj:`Dict[str, float]`, `optional`): The potential dictionary of metrics (if the dataset
              contained labels).
        """
        self._max_length = max_length
        self._num_beams = num_beams
        return super().predict(test_dataset, ignore_keys=ignore_keys, metric_key_prefix=metric_key_prefix)

    def prediction_step(
            self,
            model: nn.Module,
            inputs: Dict[str, Union[torch.Tensor, Any]],
            prediction_loss_only: bool,
            ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[float], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Perform an evaluation step on :obj:`model` using obj:`inputs`.

        Subclass and override to inject custom behavior.

        Args:
            model (:obj:`nn.Module`):
                The model to evaluate.
            inputs (:obj:`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.

                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument :obj:`labels`. Check your model's documentation for all accepted arguments.
            prediction_loss_only (:obj:`bool`):
                Whether or not to return the loss only.

        Return:
            Tuple[Optional[float], Optional[torch.Tensor], Optional[torch.Tensor]]: A tuple with the loss, logits and
            labels (each being optional).
        """

        if not self.args.predict_with_generate or prediction_loss_only:
            return super().prediction_step(
                model, inputs, prediction_loss_only=prediction_loss_only, ignore_keys=ignore_keys
            )

        has_labels = "labels" in inputs
        inputs = self._prepare_inputs(inputs)

        # XXX: adapt synced_gpus for fairscale as well


        gen_kwargs = {
            "max_length": 128,
            "num_beams": 5,
            "synced_gpus": True if is_deepspeed_zero3_enabled() else False,
        }
        # print(gen_kwargs)
        # if self.model.config.decoder_enhance == 1:
        #     encoder_parallel_classify = self.model(input_ids=inputs["input_ids"],entailment_mask = inputs['entailment_mask'],attention_mask=inputs["attention_mask"],decoder_input_ids =torch.tensor([[self.model.config.decoder_start_token_id]]),encoder_only=True).argmax(dim=-1).squeeze().tolist()
        #
        #     decoder_input = []
        #
        #     entailment_list = ['true', 'unknown', 'false']
        #     if isinstance(encoder_parallel_classify, list):
        #         for item in encoder_parallel_classify:
        #             decoder_input.append(entailment_list[item])
        #
        #     elif isinstance(encoder_parallel_classify, int):
        #         decoder_input.append(entailment_list[encoder_parallel_classify])
        #
        #     else:
        #         print("unknown type error in encoder_parallel_classify")
        #
        #     decoder_input.append('final: ')
        #
        #     decoder_input_ids = self.tokenizer((' '.join(decoder_input)),return_tensors="pt").input_ids
        #     decoder_input_ids = self.model.prepare_decoder_input_ids_from_labels(labels=decoder_input_ids)
        #
        #     generated_tokens = self.model.generate(
        #         inputs["input_ids"],
        #         attention_mask=inputs["attention_mask"],
        #         decoder_input_ids = decoder_input_ids.to(inputs["input_ids"].device),
        #         **gen_kwargs,
        #     )
        #
        # else:
        entailment_preds = torch.tensor([1] * inputs["input_ids"].shape[0]).to(inputs['input_ids'].device)
        entailment_label = torch.tensor([1]* inputs["input_ids"].shape[0]).to(inputs['input_ids'].device)
        encoder_label = torch.tensor([1] * inputs["input_ids"].shape[0]).to(inputs['input_ids'].device)
        encoder_preds = torch.tensor([1]* inputs["input_ids"].shape[0]).to(inputs['input_ids'].device)

        if "entailment_mask" in inputs.keys() and self.model.entailment_alpha > 0.0:
            entailment_preds, _ = self.model(input_ids=inputs["input_ids"],
                                                         entailment_mask=inputs['entailment_mask'],
                                                         attention_mask=inputs["attention_mask"],
                                                         encoder_label=inputs['encoder_label'],
                                                         entailment_len=inputs['entailment_len'],
                                                         edu_attention_mask =inputs['edu_attention_mask'],
                                                         rule_mask=inputs['rule_mask'], encoder_only=True)

            entailment_preds = entailment_preds.masked_fill_((~inputs['rule_mask']), -100)

            entailment_label = inputs['entailment_label']

            # encoder_label = inputs['encoder_label']

        if model.config.decoder_enhance:
            decoder_input = []

            entailment_list = ['true', 'unknown', 'false']

            entailment_iter = entailment_preds.squeeze().tolist()
            if isinstance(entailment_iter, int):
                if entailment_iter != -100:
                    decoder_input.append(entailment_list[entailment_preds])

            else:
                for item in entailment_iter:
                    if item != -100:
                        decoder_input.append(entailment_list[item])

            decoder_input.append('final: ')

            decoder_input_ids = self.tokenizer((' '.join(decoder_input)), return_tensors="pt").input_ids
            decoder_input_ids = self.model.prepare_decoder_input_ids_from_labels(labels=decoder_input_ids)

            generated_tokens = self.model.generate(
                inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                decoder_input_ids=decoder_input_ids.to(inputs["input_ids"].device),
                **gen_kwargs,
            )

        else:
            generated_tokens = self.model.generate(
                inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                **gen_kwargs,
            )

        # in case the batch is shorter than max length, the output should be padded
        if generated_tokens.shape[-1] < gen_kwargs["max_length"]:
            generated_tokens = self._pad_tensors_to_max_len(generated_tokens, gen_kwargs["max_length"])

        with torch.no_grad():
            # if self.use_amp:
            #     with autocast():
            #         outputs = model(**inputs)
            # else:
            outputs = model(**inputs)
            if has_labels:
                if self.label_smoother is not None:
                    loss = self.label_smoother(outputs, inputs["labels"]).mean().detach()
                else:
                    loss = (outputs["loss"] if isinstance(outputs, dict) else outputs[0]).mean().detach()
            else:
                loss = None

        if self.args.prediction_loss_only:
            return (loss, None, None)

        labels = inputs["labels"]
        if labels.shape[-1] < gen_kwargs["max_length"]:
            labels = self._pad_tensors_to_max_len(labels, gen_kwargs["max_length"])

        return (loss, (generated_tokens, entailment_preds, encoder_preds), (labels, entailment_label, encoder_label))

    def _pad_tensors_to_max_len(self, tensor, max_length):
        if self.tokenizer is None:
            raise ValueError(
                f"Tensor need to be padded to `max_length={max_length}` but no tokenzier was passed when creating "
                "this `Trainer`. Make sure to create your `Trainer` with the appropriate tokenizer."
            )
        # If PAD token is not defined at least EOS token has to be defined
        pad_token_id = (
            self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id
        )

        padded_tensor = pad_token_id * torch.ones(
            (tensor.shape[0], max_length), dtype=tensor.dtype, device=tensor.device
        )
        padded_tensor[:, : tensor.shape[-1]] = tensor
        return padded_tensor

    def create_optimizer(self):
        """
        Setup the optimizer.

        We provide a reasonable default that works well. If you want to use something else, you can pass a tuple in the
        Trainer's init through :obj:`optimizers`, or subclass and override this method in a subclass.
        """
        if self.optimizer is None:
            decay_parameters = get_parameter_names(self.model, [nn.LayerNorm])
            decay_parameters = [name for name in decay_parameters if "bias" not in name]
            optimizer_grouped_parameters = [
                {
                    "params": [p for n, p in self.model.named_parameters() if n in decay_parameters and "transformer" not in n],
                    "weight_decay": self.args.weight_decay,
                    "lr": self.args.learning_rate,
                },
                {
                    "params": [p for n, p in self.model.named_parameters() if n not in decay_parameters and "transformer" not in n],
                    "weight_decay": 0.0,
                    "lr": self.args.learning_rate,
                },
                {
                    "params": [p for n, p in self.model.named_parameters() if n in decay_parameters and "transformer" in n],
                    "weight_decay": self.args.weight_decay,
                    "lr": 2e-5,
                },
                {
                    "params": [p for n, p in self.model.named_parameters() if n not in decay_parameters and "transformer" in n],
                    "weight_decay": 0.0,
                    "lr": 2e-5,
                },
            ]
            optimizer_cls = Adafactor if self.args.adafactor else AdamW
            if self.args.adafactor:
                optimizer_cls = Adafactor
                optimizer_kwargs = {"scale_parameter": False, "relative_step": False}
            else:
                optimizer_cls = AdamW
                optimizer_kwargs = {
                    "betas": (self.args.adam_beta1, self.args.adam_beta2),
                    "eps": self.args.adam_epsilon,
                }
            # optimizer_kwargs["lr"] = self.args.learning_rate
            if self.sharded_ddp == ShardedDDPOption.SIMPLE:
                self.optimizer = OSS(
                    params=optimizer_grouped_parameters,
                    optim=optimizer_cls,
                    **optimizer_kwargs,
                )
            else:
                self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)

        if is_sagemaker_mp_enabled():
            self.optimizer = smp.DistributedOptimizer(self.optimizer)

        return self.optimizer