alphas="9 42 110 123 3120 "
for alpha in $alphas
do
datasets="dev test "
for dataset in $datasets
    do
    echo $alpha
    echo $dataset  

    # pre="/home/t-zhangxiao/DeNoiseOcmrc/ablation/large-2e-4-base-multi-et0.9-multisetUnseen-candidates-5-seed-${alpha}/answer_${dataset}.json"
    # gold="/home/t-zhangxiao/DeNoiseOcmrc/ablation/large-2e-4-base-multi-et0.9-multisetUnseen-candidates-5-seed-${alpha}/golden_${dataset}.json"
    # python3 oEvaluator.py $pre $gold 
    # python3 oEvaluator.py $pre $gold > large-2e-4-base-multi-et0.9-multisetUnseen-candidates-5-seed-${alpha}${dataset} 2>&1
    # echo $pre
    # echo $gold


    # pre="/home/t-zhangxiao/DeNoiseOcmrc/ablation/large-2e-4-base-multi-et0.9-multisetUnseen-${alpha}/answer_${dataset}.json"
    # gold="/home/t-zhangxiao/DeNoiseOcmrc/ablation/large-2e-4-base-multi-et0.9-multisetUnseen-${alpha}/golden_${dataset}.json"
    # python3 oEvaluator.py $pre $gold 
    # python3 oEvaluator.py $pre $gold > large-2e-4-base-multi-et0.9-multisetUnseen-${alpha}${dataset} 2>&1
    # echo $pre
    # echo $gold

    # pre="/home/t-zhangxiao/DeNoiseOcmrc/ablation/large-2e-4-base-multi-et0.0-multisetUnseen-candidates-5-seed-${alpha}/answer_${dataset}.json"
    # gold="/home/t-zhangxiao/DeNoiseOcmrc/ablation/large-2e-4-base-multi-et0.0-multisetUnseen-candidates-5-seed-${alpha}/golden_${dataset}.json"
    # python3 oEvaluator.py $pre $gold 
    # python3 oEvaluator.py $pre $gold > large-2e-4-base-multi-et0.0-multisetUnseen-candidates-5-seed-${alpha}${dataset} 2>&1
    # echo $pre
    # echo $gold

    pre="/home/t-zhangxiao/DeNoiseOcmrc/ablation/large-2e-4-base-multi-et0.0-multisetUnseen-candidates-1-seed-${alpha}-noshuffle/answer_${dataset}.json"
    gold="/home/t-zhangxiao/DeNoiseOcmrc/ablation/large-2e-4-base-multi-et0.0-multisetUnseen-candidates-1-seed-${alpha}-noshuffle/golden_${dataset}.json"
    python3 oEvaluator.py $pre $gold 
    python3 oEvaluator.py $pre $gold > large-2e-4-base-multi-et0.0-multisetUnseen-candidates-1-seed-noshuffle-${alpha}${dataset} 2>&1
    echo $pre
    echo $gold
    
    done
done