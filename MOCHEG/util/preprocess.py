import os
import pandas as pd
import random
from sentence_transformers import SentenceTransformer, util
import torch
from tqdm import tqdm

model_path_sbert = 'sbert-path'
model = SentenceTransformer(model_path_sbert)


def get_most_dissimilar_evidences(current_evidence, retrieval_evidence, top_k=5):
    current_evidence = [str(evidence) if isinstance(evidence, float) else evidence for evidence in current_evidence]
    current_evidence_embeddings = model.encode(current_evidence, convert_to_tensor=True)
    retrieval_evidence_embeddings = model.encode(retrieval_evidence, convert_to_tensor=True)

    max_similarity_scores = []

    for retrieval_doc_embedding in retrieval_evidence_embeddings:
        cosine_scores = util.pytorch_cos_sim(retrieval_doc_embedding, current_evidence_embeddings)
        max_score = torch.max(cosine_scores).item()
        max_similarity_scores.append(max_score)

    dissimilar_indices = torch.argsort(torch.tensor(max_similarity_scores), descending=False)[:top_k].tolist()

    dissimilar_evidences = [retrieval_evidence[i] for i in dissimilar_indices]

    return dissimilar_evidences


def merge_evidence(data_path, out_name, drop_invalid_data_func):
    corpus = os.path.join(data_path, "Corpus2.csv")
    # print(data_path)
    retrieval_result_path = os.path.join(data_path, "retrieval/retrieval_result.csv")
    # print(retrieval_result_path)
    retrieval_evidence_df = pd.read_csv(retrieval_result_path)

    evidence_df = pd.read_csv(corpus, encoding="utf8")
    evidence_df = drop_invalid_data_func(evidence_df)
    out_corpus = os.path.join(data_path, out_name)
    evidence_list = []
    # all_evidences_list = evidence_df["Evidence"].tolist()
    confusing_evidence = []
    random_evidence = []
    evidence_merged_df = evidence_df.drop_duplicates(subset='claim_id', keep="first")
    # all_evidences_list =

    evidence_merged_df = evidence_merged_df.drop(columns=['Evidence'])
    evidence_merged_df = evidence_merged_df.reset_index(drop=True)

    ## retrieval evidence
    # evidence_retrieval = pd.read_csv(os.path.join(data_path,"retrieval_result.csv") ,encoding="utf8")

    for i, row in tqdm(evidence_merged_df.iterrows()):
        claim_id = row["claim_id"]

        ###
        # found = evidence_retrieval.loc[evidence_retrieval['claim_id'] ==claim_id]
        ###
        found = evidence_df.loc[evidence_df['claim_id'] == claim_id]
        evidence = append_str(found["Evidence"].values)
        evidence = evidence.replace("<p>", "")
        evidence = evidence.replace("</p>", "")
        evidence_list.append(evidence)

        # c_evidence = random.sample(all_evidences_list,5)
        system_evidence_found = retrieval_evidence_df.loc[retrieval_evidence_df['claim_id'].astype(int) == int(claim_id)]
        # print(len(list(system_evidence_found["Evidence"].values)))
        system_evidence = list(system_evidence_found["Evidence"].values)[0].strip().split("[SEP_CCMC]")

        system_evidence = [s for s in system_evidence if s.strip()]
        # print(system_evidence)
        c_evidence = get_most_dissimilar_evidences(list(found["Evidence"].values),system_evidence,5)
        # print(c_evidence)
        

        r_evidence = append_str(c_evidence.copy())
        r_evidence = r_evidence.replace("<p>", "")
        r_evidence = r_evidence.replace("</p>", "")
        random_evidence.append(r_evidence)

        # c_evidence = append_str(list(found["Evidence"].values) + c_evidence)
        c_evidence = append_str_confusing(list(found["Evidence"].values), c_evidence)
        c_evidence = c_evidence.replace("<p>", "")
        c_evidence = c_evidence.replace("</p>", "")
        confusing_evidence.append(c_evidence)

        # print(evidence)
    evidence_merged_df.insert(13, "Evidence", evidence_list)
    evidence_merged_df.insert(14, "Confusing_Evidence", confusing_evidence)
    evidence_merged_df.insert(15, "Random_Evidence", random_evidence)
    evidence_merged_df.to_csv(out_corpus, index=False)


def append_str(evidence_array):
    evidence = ""
    for i in range(len(evidence_array)):
        evidence += str(evidence_array[i]) + " "
    # if evidence=="" or evidence==" ":
    #     print("no text evidence")
    return evidence


def append_str_confusing(evidence_array, confusing_array):
    evidence = ""
    if len(evidence_array) < 1:
        # print("no text evidence")
        return evidence
    return append_str(evidence_array + confusing_array)



