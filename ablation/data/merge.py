
import jsonlines
import json


def getEvalData(dataset,path):
    answer = []
    data = []

    with open(f"{path}{dataset}_generated_predictions.txt") as file:
        line = file.readline()
        while line:
            answer.append(line.split("final: ")[-1].strip().strip("\n").capitalize())
            line = file.readline()
            print(answer[-1])

    golden = []
    count = 0
    with jsonlines.open(f"open_retrieval_sharc_{dataset}.json") as reader,jsonlines.open(f"{dataset}_answer.json","w") as writer:
        for item in reader:
            print(count)
            golden.append(item)
            temp = {"utterance_id":item["utterance_id"],"answer": answer[count],"snippet_seen":item["snippet_seen"]}
            data.append(temp)
            writer.write(temp)
            count+=1

    unseen_predict = []
    seen_predict = []

    unseen_golden = []
    seen_golden = []

    for i in range(len(golden)):
        if golden[i]["snippet_seen"]:
            seen_golden.append(golden[i])
            seen_predict.append(data[i])
        else:
            unseen_golden.append(golden[i])
            unseen_predict.append(data[i])

    with open(f"{path}golden_{dataset}.json","w") as file:
        json.dump(golden,file)
    with open(f"{path}answer_{dataset}.json","w") as file:
        json.dump(data,file)
            

    with open(f"{path}golden_{dataset}_unseen.json","w") as file:
        json.dump(unseen_golden,file)
    with open(f"{path}answer_{dataset}_unseen.json","w") as file:
        json.dump(unseen_predict,file)
            
    with open(f"{path}golden_{dataset}_seen.json","w") as file:
        json.dump(seen_golden,file)
    with open(f"{path}answer_{dataset}_seen.json","w") as file:
        json.dump(seen_predict,file)

for alpha in[0.9]:
    print(alpha)
    for dataset in ["dev","test"]:
        print(dataset)
        for seed in [9,123,110,42,3120]:
            # path1=f"/home/t-zhangxiao/DeNoiseOcmrc/ablation/large-2e-4-base-multi-et0.9-multisetUnseen-candidates-5-seed-{seed}/"
            # getEvalData(dataset,path1)
            # path2=f"/home/t-zhangxiao/DeNoiseOcmrc/ablation/large-2e-4-base-multi-et0.9-multisetUnseen-{seed}/"
            # getEvalData(dataset,path2)      
            # path3=f"/home/t-zhangxiao/DeNoiseOcmrc/ablation/large-2e-4-base-multi-et0.0-multisetUnseen-candidates-5-seed-{seed}/"
            # getEvalData(dataset,path3)
            path0=f"/home/t-zhangxiao/DeNoiseOcmrc/ablation/large-2e-4-base-multi-et0.0-multisetUnseen-candidates-1-seed-{seed}-noshuffle/"
            getEvalData(dataset,path0)       