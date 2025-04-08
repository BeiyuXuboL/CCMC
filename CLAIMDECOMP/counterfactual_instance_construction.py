import json
import random

datapath = "data/train-site-restricted-wo-search-results.jsonl"
save_path = "data/train-data-w-doc.jsonl"
def counterfactual_instance_construct(data_path):
    # merge all sentences retrieved by each claim according to the retrieval
    # scores
    with open(data_path, 'r') as file:
        data = [json.loads(line) for line in file]
    # totalnum = len(data)
    res = []
    for i,item in enumerate(data):
        # res_one = {}
        res_one = item.copy()
        while True:
            refenence_item = random.choice(data)
            if res_one["example_id"] != refenence_item["example_id"]:
                break
        res_one["summary"] = item["summarization_prompt"]
        res_one["noise_evidence"] = refenence_item["summarization_prompt"]
        res_one["confuse_evidence"] = item["summarization_prompt"] + " " + refenence_item["summarization_prompt"]
        res.append(res_one)
    with open(save_path, 'w') as file:
        for d in res:
            json.dump(d, file)

counterfactual_instance_construct(data_path=datapath)


datapath = "data/dev-site-restricted-wo-search-results.jsonl"
save_path = "data/dev-data-w-doc.jsonl"
def counterfactual_instance_construct1(data_path):
    # merge all sentences retrieved by each claim according to the retrieval
    # scores
    with open(data_path, 'r') as file:
        data = [json.loads(line) for line in file]
    # totalnum = len(data)
    res = []
    for i,item in enumerate(data):
        # res_one = {}
        res_one = item.copy()
        # while True:
        #     refenence_item = random.choice(data)
        #     if res_one["example_id"] != refenence_item["example_id"]:
        #         break
        res_one["summary"] = item["summarization_prompt"]
        res_one["noise_evidence"] = item["summarization_prompt"]
        res_one["confuse_evidence"] = item["summarization_prompt"]
        res.append(res_one)
    with open(save_path, 'w') as file:
        for d in res:
            json.dump(d, file)

counterfactual_instance_construct1(data_path=datapath)


datapath = "data/test-site-restricted-wo-search-results.jsonl"
save_path = "data/test-data-w-doc.jsonl"
def counterfactual_instance_construct2(data_path):
    # merge all sentences retrieved by each claim according to the retrieval
    # scores
    with open(data_path, 'r') as file:
        data = [json.loads(line) for line in file]
    # totalnum = len(data)
    res = []
    for i,item in enumerate(data):
        # res_one = {}
        res_one = item.copy()
        # while True:
        #     refenence_item = random.choice(data)
        #     if res_one["example_id"] != refenence_item["example_id"]:
        #         break
        res_one["summary"] = item["summarization_prompt"]
        res_one["noise_evidence"] = item["summarization_prompt"]
        res_one["confuse_evidence"] = item["summarization_prompt"]
        res.append(res_one)
    with open(save_path, 'w') as file:
        for d in res:
            json.dump(d, file)

counterfactual_instance_construct2(data_path=datapath)