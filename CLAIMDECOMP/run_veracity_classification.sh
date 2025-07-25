#!/usr/bin/env bash
model_name_path="microsoft/deberta-large"
classifier="models/hf_classifier.py"
train_file="./data/train-data-w-doc.jsonl"
validation_file="./data/dev-data.jsonl"
test_file="./data/test-data-w-doc.jsonl"
output_dir="./model_output_system_w_doc"
eval_steps=100
max_seq_length=1024
train_batch_size=2
eval_batch_size=2
epochs=25
learning_rate=3e-5
gradient_accumulation_steps=2
use_claim=true
use_justification=false
use_decomposed_question=false
use_bm25_retrieval=false
use_answer=false
use_qa_pairs=false
use_gpt_rationale=true
metric_for_best_model="MAE"
gpu_num=4
topk_docs=1
topk_sents=10
seed=666666
num_class=6

while [ $# -gt 0 ]; do
   if [[ $1 == *"--"* ]]; then
        param="${1/--/}"
        declare $param="$2"
   fi
  shift
done

python -m torch.distributed.launch --nproc_per_node ${gpu_num} --master_port 1235 ${classifier} \
--model_name_or_path ${model_name_path} \
--cache_dir "./cache" \
--train_file ${train_file} \
--validation_file ${validation_file} \
--test_file ${test_file} \
--do_train true \
--do_eval true \
--do_predict true \
--logging_strategy steps \
--logging_first_step true \
--logging_steps 100 \
--evaluation_strategy steps \
--max_seq_length ${max_seq_length} \
--eval_steps ${eval_steps} \
--save_strategy steps \
--save_steps ${eval_steps} \
--save_total_limit 2 \
--metric_for_best_model ${metric_for_best_model} \
--load_best_model_at_end true \
--greater_is_better false \
--gradient_accumulation_steps ${gradient_accumulation_steps} \
--num_train_epochs ${epochs} \
--learning_rate ${learning_rate} \
--output_dir ${output_dir} \
--overwrite_output_dir true \
--per_device_train_batch_size ${train_batch_size} \
--per_device_eval_batch_size ${eval_batch_size} \
--report_to none \
--seed ${seed} \
--use_justification ${use_justification} \
--dataset_name "ClaimDecomp" \
--use_claim ${use_claim} \
--use_decomposed_question ${use_decomposed_question} \
--use_answer ${use_answer} \
--use_bm25_retrieval ${use_bm25_retrieval} \
--use_qa_pairs ${use_qa_pairs} \
--topk_docs ${topk_docs} \
--topk_sents ${topk_sents} \
--num_class ${num_class} \
--use_gpt_rationale ${use_gpt_rationale}

