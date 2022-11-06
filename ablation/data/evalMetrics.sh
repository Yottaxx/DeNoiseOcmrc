alphas="0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 "
for alpha in $alphas
do
datasets="dev test "
for dataset in $datasets
    do
    echo $alpha
    echo $dataset
    pre="/home/t-zhangxiao/DeNoiseOcmrc/ablation/large-2e-4-base-multi-et${alpha}-multisetUnseen/answer_${dataset}.json"
    gold="/home/t-zhangxiao/DeNoiseOcmrc/ablation/large-2e-4-base-multi-et${alpha}-multisetUnseen/golden_${dataset}.json"
    python3 oEvaluator.py $pre $gold > ${alpha}${dataset} 2>&1
    done
done