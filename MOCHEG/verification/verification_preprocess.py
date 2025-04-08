

from retrieval.utils.news import News
from util.preprocess import merge_evidence
import random
import os
import pandas as pd

from util.read_example import read_image

def drop_invalid_data_for_verification(df):
    return df
    
    
def preprocess_for_verification_one_subset(data_path):
    out_name="Corpus2_for_verification.csv"
    merge_evidence( data_path,out_name,drop_invalid_data_for_verification)
    clean_text_evidence( data_path,out_name) 
    add_image_evidence(data_path,out_name)


def clean_text_evidence(data_path,out_name):
    corpus=os.path.join(data_path,out_name)
    # /retrieval
    evidence_list=[]
    confusing_evidence_list=[]
    evidence_df = pd.read_csv(corpus ,encoding="utf8")  
    for i,row in evidence_df.iterrows():
        evidence=row["Evidence"]
        confusing_evidence = row["Confusing_Evidence"]
        ruling_article=row["Origin"]
        ruling_outline=row["ruling_outline"]
        if pd.isna(evidence) or len(evidence)<=0:
            if pd.isna(ruling_outline) or len(ruling_outline)<=0:
                evidence=ruling_article #use ruling article to be evidence
            else:
                evidence=ruling_outline
        if pd.isna(confusing_evidence) or len(confusing_evidence)<=0:
            if pd.isna(ruling_outline) or len(ruling_outline)<=0:
                confusing_evidence=ruling_article #use ruling article to be evidence
            else:
                confusing_evidence=ruling_outline
        evidence_list.append(evidence)
        confusing_evidence_list.append(confusing_evidence)
    evidence_df =evidence_df.drop(columns=['Evidence'])
    evidence_df.insert(13, "Evidence",evidence_list )
    evidence_df =evidence_df.drop(columns=['Confusing_Evidence'])
    evidence_df.insert(14, "Confusing_Evidence",confusing_evidence_list )
    evidence_df.to_csv(corpus,index=False)
 


 
def add_image_evidence(data_path,out_name):
    news_dict={}
    news_dict,_,all_images_list=read_image(data_path,news_dict,content="img")
    corpus=os.path.join(data_path,out_name)
    img_evidence_str_list=[]
    confusing_img_evidence_str_list=[]
    random_img_evidence_str_list=[]
    evidence_df = pd.read_csv(corpus ,encoding="utf8")

    retrieval_result_path = os.path.join(data_path, "retrieval/retrieval_result.csv")
    retrieval_evidence_df = pd.read_csv(retrieval_result_path)

    for i,row in evidence_df.iterrows():
        claim_id=row["claim_id"]

        system_evidence_found = retrieval_evidence_df.loc[retrieval_evidence_df['claim_id'].astype(int) == int(claim_id)]
        random_image_evidences = list(system_evidence_found["img_evidences"].values)[0].split(";")

        random_img_evidence_str=";".join(random_image_evidences[:5])
        if claim_id in news_dict.keys():
            gold_img_evidence_list=news_dict[claim_id].get_img_evidence_list()
            gold_img_evidence_str=";".join(gold_img_evidence_list)
            top_k_random_img_evidence = []
            for system_img_evidence in random_image_evidences:
                if system_img_evidence in gold_img_evidence_list:
                    continue
                top_k_random_img_evidence.append(system_img_evidence)
                if len(top_k_random_img_evidence) >= 5:
                    break
            random_img_evidence_str = ";".join(top_k_random_img_evidence)
            if len(gold_img_evidence_list) == 0:
                confusing_img_evidence_str=""
            else:
                confusing_img_evidence_list = gold_img_evidence_list + top_k_random_img_evidence
                random.shuffle(confusing_img_evidence_list)
                confusing_img_evidence_str=";".join(confusing_img_evidence_list) 
        else:
            gold_img_evidence_str=""
            confusing_img_evidence_str=""
        img_evidence_str_list.append(gold_img_evidence_str)
        confusing_img_evidence_str_list.append(confusing_img_evidence_str)
        random_img_evidence_str_list.append(random_img_evidence_str)
    evidence_df.insert(16, "img_evidences",img_evidence_str_list )
    evidence_df.insert(17, "confusing_img_evidences",confusing_img_evidence_str_list )
    evidence_df.insert(18, "random_img_evidences",random_img_evidence_str_list )
    evidence_df.to_csv(corpus,index=False)
     


def preprocess_for_verification(data_path):
    preprocess_for_verification_one_subset( data_path+ "/train")
    print("finish one")
    preprocess_for_verification_one_subset( data_path+"/val")
    print("finish one")
    preprocess_for_verification_one_subset(  data_path+"/test")
        
        
        
import argparse
def parser_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path',type=str,help=" ",default=" ")
    args = parser.parse_args()
    return args
 



if __name__ == '__main__':
    args = parser_args()
    preprocess_for_verification_one_subset( args.data_path+ "/train")