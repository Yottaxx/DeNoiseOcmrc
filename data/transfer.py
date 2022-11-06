import copy
import json
import random

import jsonlines
from tqdm import tqdm

with open("./id2snippet.json") as file:
    id2snippet = json.load(file)

seen = set()
unseen = set()
for dataset in ["dev","test","train"]:
    with jsonlines.open("open_retrieval_sharc_{}.json".format(dataset)) as reader:
        for item in reader:
            if dataset == "train":
                seen.add(item["gold_snippet_id"])
                continue

            if item["snippet_seen"]:
                seen.add(item["gold_snippet_id"])
            else:
                unseen.add(item["gold_snippet_id"])

# TODO : Shuffle
for dataset in ["dev", "test","train"]:
    data = []
    isTrain = (dataset == "train")
    breakLen = 5
    if isTrain:
        breakLen = 20
    matches = 0
    with jsonlines.open("./retrieved/npSetContinue/open_retrieval_sharc_{}_retrieved.json".format(dataset)) as reader:
        for item in tqdm(reader):
            itemList = []
            trueFlag = False
            usedIds = set()
            retrievalIds = item['retrievalIds']

            if isTrain:
                positiveCtx = id2snippet[item['gold_snippet_id']]
                usedIds.add(item['gold_snippet_id'])
                positive = copy.deepcopy(item)
                positive['retrieval'] = None
                positive['retrievalIds'] = None
                positive['label'] = 1
                positive['snippet'] = positiveCtx
                positive['snippet_id'] = item['gold_snippet_id']
                positive['utterance_id'] = item['utterance_id']
                itemList.append(positive)
                trueFlag = True

            for ids in retrievalIds:
                if ids == item['gold_snippet_id'] and isTrain:
                    continue
                if ids in unseen and isTrain:
                    continue
                if len(itemList) == breakLen:
                    break
                else:
                    # negative = copy.deepcopy(positive)
                    negative = copy.deepcopy(item)
                    negative['retrieval'] = None
                    negative['retrievalIds'] = None

                    negative['snippet'] = id2snippet[ids]

                    if ids == item['gold_snippet_id']:
                        negative['label'] = 1
                        trueFlag = True
                        matches+=1
                    else:
                        negative['label'] = 0

                    negative['snippet_id'] = ids
                    # negative['answer'] = "Irrelevant"
                    itemList.append(negative)
                    usedIds.add(ids)
            if not trueFlag:
                itemList[-1]["label"] = 1
                trueFlag = True
            assert trueFlag
            assert len(itemList) == breakLen or isTrain, "find five negatives error {}".format(dataset)

            if isTrain:
                idsLists = list(id2snippet.keys())
                random.shuffle(idsLists)
                for ids in idsLists:
                    if ids in unseen:
                        continue
                    if len(itemList) == 50:
                        break
                    if ids in usedIds:
                        continue
                    negative = copy.deepcopy(item)
                    negative['retrieval'] = None
                    negative['retrievalIds'] = None

                    negative['snippet'] = id2snippet[ids]

                    if ids == item['gold_snippet_id']:
                        negative['label'] = 1
                        trueFlag = True
                    else:
                        negative['label'] = 0

                    negative['snippet_id'] = ids
                    # negative['answer'] = "Irrelevant"
                    itemList.append(negative)
                    usedIds.add(ids)
            # if not isTrain:
            #     random.shuffle(itemList)
            data.extend(itemList)
    print(matches)
    with open("./toEntailmenProcess/sharc_{}.json".format(dataset), "w") as file:
        json.dump(data, file, ensure_ascii=False)
