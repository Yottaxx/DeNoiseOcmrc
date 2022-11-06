alphas="9 42 110 123 3120 "
for alpha in $alphas
do
datasets="test "
for dataset in $datasets
    do
    echo $alpha
    echo $dataset
    # pre="/home/t-zhangxiao/DeNoiseOcmrc/ablation/large-2e-4-base-multi-et0.9-multisetUnseen-${alpha}/answer_${dataset}_seen.json"
    # gold="/home/t-zhangxiao/DeNoiseOcmrc/ablation/large-2e-4-base-multi-et0.9-multisetUnseen-${alpha}/golden_${dataset}_seen.json"

    # echo $pre
    # echo $gold
    # # pre="/home/t-zhangxiao/DeNoiseOcmrc/ablation/large-2e-4-base-multi-et${alpha}-multisetUnseen/answer_${dataset}_unseen.json"
    # # gold="/home/t-zhangxiao/DeNoiseOcmrc/ablation/large-2e-4-base-multi-et${alpha}-multisetUnseen/golden_${dataset}_unseen.json"
    # python3 oEvaluator.py $pre $gold 
    # python3 oEvaluator.py $pre $gold > large-2e-4-base-multi-et0.9-multisetUnseen-${alpha}_seen${dataset} 2>&1


    pre="/home/t-zhangxiao/DeNoiseOcmrc/ablation/large-2e-4-base-multi-et0.0-multisetUnseen-candidates-5-seed-${alpha}-noshuffle/answer_${dataset}_seen.json"
    gold="/home/t-zhangxiao/DeNoiseOcmrc/ablation/large-2e-4-base-multi-et0.0-multisetUnseen-candidates-5-seed-${alpha}-noshuffle/golden_${dataset}_seen.json"
    python3 oEvaluator.py $pre $gold 
    python3 oEvaluator.py $pre $gold > large-2e-4-base-multi-et0.0-multisetUnseen-candidates-5-seed-noshuffle-${alpha}_seen${dataset} 2>&1
    echo $pre
    echo $gold

    done
done