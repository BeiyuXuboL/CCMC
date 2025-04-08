import json
import logging
import os
import datasets
import transformers
import sys
import numpy as np
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    HfArgumentParser,
    TrainingArguments,
    set_seed,
    EvalPrediction,
    EarlyStoppingCallback
)
from transformers.trainer_utils import get_last_checkpoint
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error
from utils.other_arguments import DataArguments, ModelArguments
from utils.preprocessing import concat_claim_retrieved_sents, feed_claim_with_context, \
concat_claim_justification, concat_claim_qa_pairs, concat_claim_questions, \
concat_claim_gpt_rationale, concat_claim_gpt_rationale2, concat_claim_gpt_rationale3
import torch
import torch.nn as nn
import random

logger = logging.getLogger(__name__)
SEP_TK = "[SEP]"


class MultiView(torch.nn.Module):
    def __init__(self, view1_model_path, other_view_model_path, model_args):
        """
        In the constructor we instantiate two nn.Linear modules and assign them as
        member variables.
        """
        super(MultiView, self).__init__()
        
        config1 = AutoConfig.from_pretrained(
        view1_model_path,
        num_labels=6,
        # finetuning_task=data_args.task_name,
        cache_dir=model_args.cache_dir,
        )
        # self.tokenizer1 = AutoTokenizer.from_pretrained(
        #     view1_model_path.model_name_or_path,
        #     cache_dir=model_args.cache_dir,
        #     use_fast=model_args.use_fast_tokenizer,
        # )
        self.model1 = AutoModelForSequenceClassification.from_pretrained(
            view1_model_path,
            config=config1,
            cache_dir=model_args.cache_dir,
        )
        self.model1.requires_grad_(False)

     

        config2 = AutoConfig.from_pretrained(
        other_view_model_path,
        num_labels=6,
        # finetuning_task=data_args.task_name,
        cache_dir=model_args.cache_dir,
        )
        # self.tokenizer2 = AutoTokenizer.from_pretrained(
        #     view1_model_path.model_name_or_path,
        #     cache_dir=model_args.cache_dir,
        #     use_fast=model_args.use_fast_tokenizer,
        # )
        self.model2 = AutoModelForSequenceClassification.from_pretrained(
            other_view_model_path,
            config=config2,
            cache_dir=model_args.cache_dir,
        )

        config3 = AutoConfig.from_pretrained(
        other_view_model_path,
        num_labels=6,
        # finetuning_task=data_args.task_name,
        cache_dir=model_args.cache_dir,
        )
        # self.tokenizer3 = AutoTokenizer.from_pretrained(
        #     view1_model_path.model_name_or_path,
        #     cache_dir=model_args.cache_dir,
        #     use_fast=model_args.use_fast_tokenizer,
        # )
        self.model3 = AutoModelForSequenceClassification.from_pretrained(
            other_view_model_path,
            config=config3,
            cache_dir=model_args.cache_dir,
        )
        self.criterion = nn.CrossEntropyLoss()
    
    def forward(self, input_ids, attention_mask, input_ids2, attention_mask2, input_ids3, attention_mask3, labels):
        """
        In the forward function we accept a Tensor of input data and we must return
        a Tensor of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Tensors.
        """
        output_G = self.model1(input_ids=input_ids, attention_mask=attention_mask)['logits']
        loss = self.criterion(output_G, labels)
        # return {"loss": loss, "logits": output_G}
        output_C = self.model2(input_ids=input_ids2, attention_mask=attention_mask2)['logits']
        output_ = self.model2(input_ids=input_ids3, attention_mask=attention_mask3)['logits']
        output_R = self.model3(input_ids=input_ids3, attention_mask=attention_mask3)['logits']
        # print(output_G)
        # loss1 = self.criterion(output_G, label)
        # loss3 = self.criterion(output_G, labels)
        loss1 = self.criterion(output_C - output_, labels)
        loss2 = self.criterion(output_R, labels)
        loss = loss + loss1 + loss2
        # output_C = self.view2(self.view1.view1.has_evidence(has_image),claim_embed,txt_evidences_embed,img_evidences_embed,has_image,claim_mask,txt_evidence_mask,img_evidence_mask)
        # # has_image, claim_embed, txt_evidences_embed,img_evidences_embed,has_image,claim_mask,txt_evidence_mask,img_evidence_mask = self.view1.view1.data_processing(claim_img_encod=random_claim_img_encod,device=device)
        # output_R = self.view3(self.view1.view1.has_evidence(has_image),claim_embed,txt_evidences_embed,img_evidences_embed,has_image,claim_mask,txt_evidence_mask,img_evidence_mask)
        # return output_G, output_C, output_R, output_C
        # output_ = self.view2(self.view1.view1.has_evidence(has_image),claim_embed,txt_evidences_embed,img_evidences_embed,has_image,claim_mask,txt_evidence_mask,img_evidence_mask)
        logits = self.view_fusion(output_G, output_C, output_R)
        # return output_G, output_C - output_, output_R, output_C
        return {"loss": loss, "logits": logits}
        # output_G - output_R, outpot_C - output_R
        # return output_G, outpot_C, output_G - output_R, outpot_C - output_R
    
    # def loss_func(self, logits, label):
    #     criterion = nn.CrossEntropyLoss()
    
    # def inference(self, input):
    #     output_G, _, output_R, output_C = self.forward(input.copy(),input.copy(),input.copy())
    #     # return output_G
    #     return self.view_fusion(output_G, output_C, output_R)
    
    # def inference_ours(self, claim_img_encod,device):
    #     output_G, _, output_R, output_C = self.forward(claim_img_encod=claim_img_encod.copy(),confusing_claim_img_encod=claim_img_encod.copy(),random_claim_img_encod=claim_img_encod.copy(),device=device)
    #     # return output_G
    #     return self.view_fusion(output_G, output_C, output_R)
    
    # def case_inference(self, claim_img_encod,device):
    #     output_G, _, output_R, output_C = self.forward(claim_img_encod=claim_img_encod.copy(),confusing_claim_img_encod=claim_img_encod.copy(),random_claim_img_encod=claim_img_encod.copy(),device=device)
    #     return output_G, output_G + output_C, output_G - output_R, self.view_fusion(output_G, output_C, output_R)
    #     # return self.view_fusion(output_G, output_C, output_R)
    
    # mean fusion
    def view_fusion(self,view1,view2,view3):
        return view1 - view3/3 + view2/3

def compute_soft_acc(y_true, y_pred):
    correct = 0
    for t, p in zip(y_true, y_pred):
        if p == 0 and t in [0, 1]:
            correct += 1
        elif p == 1 and t in [0, 1, 2]:
            correct += 1
        elif p == 2 and t in [1, 2, 3]:
            correct += 1
        elif p == 3 and t in [2, 3, 4]:
            correct += 1
        elif p == 4 and t in [3, 4, 5]:
            correct += 1
        elif p == 5 and t in [4, 5]:
            correct += 1
    return correct / len(y_true)


def compute_metrics(p: EvalPrediction):
    preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
    preds = np.argmax(preds, axis=1)
    mae = mean_absolute_error(y_true=p.label_ids, y_pred=preds)
    accuracy = accuracy_score(y_true=p.label_ids, y_pred=preds)
    soft_acc = compute_soft_acc(y_true=p.label_ids, y_pred=preds)
    f1_marco = f1_score(y_true=p.label_ids, y_pred=preds, average='macro')
    f1_micro = f1_score(y_true=p.label_ids, y_pred=preds, average='micro')

    return {'MAE': mae,
            'Accuracy': accuracy,
            "soft-acc": soft_acc,
            'f1-macro': f1_marco,
            'f1-micro': f1_micro,
            "avg": np.mean([mae, accuracy, f1_marco, f1_micro])}


def main():
    # Get args
    # training args denotes the HF training args
    # args denotes the arguments defined in the Salesforce config files
    parser = HfArgumentParser(
        (TrainingArguments, DataArguments, ModelArguments))
    training_args, data_args, model_args = parser.parse_args_into_dataclasses()
    seed = random.randint(1,109999999)
    set_seed(seed)
    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # load datasets
    data_files = {}
    if training_args.do_train:
        data_files["train"] = data_args.train_file
    if training_args.do_eval:
        data_files['validation'] = data_args.validation_file
    if training_args.do_predict:
        data_files['test'] = data_args.test_file

    raw_datasets = datasets.load_dataset(
        "json",
        data_files=data_files,
        cache_dir=model_args.cache_dir,
    )

    if data_args.num_class == 6:
        label_classes = ['pants-fire', 'false', 'barely-true', 'half-true', 'mostly-true', 'true']
        num_labels = len(label_classes)
        label_to_id = {v: i for i, v in enumerate(label_classes)}
    else:
        label_classes = ['pants-fire', 'false', 'barely-true', 'half-true', 'mostly-true', 'true']
        num_labels = len(label_classes) // 2
        label_to_id = {v: (i // 2) for i, v in enumerate(label_classes)}
    # Load pretrained model and tokenizer
    #
    # In distributed training, the .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    # config = AutoConfig.from_pretrained(
    #     model_args.config_name if model_args.config_name else model_args.model_name_or_path,
    #     num_labels=num_labels,
    #     # finetuning_task=data_args.task_name,
    #     cache_dir=model_args.cache_dir,
    # )
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast_tokenizer,
    )
    # model = AutoModelForSequenceClassification.from_pretrained(
    #     model_args.model_name_or_path,
    #     config=config,
    #     cache_dir=model_args.cache_dir,
    # )
    model_name_path_other_views="microsoft/deberta-large"
    model = MultiView(model_args.model_name_or_path, model_name_path_other_views, model_args)

    # ablation
    # check_path = "xxx"
    # weights = torch.load(check_path)
    # model.load_state_dict(weights)

    def add_decomp_questions(text, decomp):
        for q in decomp:
            text += " " + q.strip()
        return text

    def add_answers(text, ans):
        for a in ans:
            text += " " + a.strip()
        return text

    def convert_raw_to_indices(examples):
        if data_args.dataset_name == "liar-plus":
            if model_args.use_justification:
                input_text = []
                for claim, just in zip(examples['claim'], examples['justification']):
                    if not just:
                        just = ""
                    input_text.append((claim + f" {SEP_TK} " + just).strip())
            else:
                input_text = examples['claim']
        elif data_args.dataset_name == "ClaimDecomp":
            input_text = []
            input_text1 = []
            input_text2 = []
            input_text3 = []
            if model_args.use_claim:
                input_text = feed_claim_with_context(examples)
                # print(examples.keys())
                # exit(0)
                if model_args.use_bm25_retrieval:
                    logging.info("Using bm25 retrieved evidence")
                    input_text = concat_claim_retrieved_sents(
                        input_text,
                        examples,
                        topk_sents=data_args.topk_sents,
                        topk_docs=data_args.topk_docs
                    )
                elif model_args.use_justification:
                    logging.info("Using claim and justification concat")
                    input_text = concat_claim_justification(
                        input_text,
                        examples
                    )
                elif model_args.use_gpt_rationale:
                    logging.info("Using claim and GPT summarization concat")
                    input_text1 = concat_claim_gpt_rationale(
                        input_text,
                        examples
                    )
                    input_text2 = concat_claim_gpt_rationale2(
                        input_text,
                        examples
                    )
                    input_text3 = concat_claim_gpt_rationale3(
                        input_text,
                        examples
                    )
                elif model_args.use_qa_pairs:
                    logging.info("Using claim and QA pairs concat")
                    input_text = concat_claim_qa_pairs(
                        input_text,
                        examples
                    )
                elif model_args.use_generated_questions:
                    logging.info("Using claim and QA pairs concat")
                    input_text = concat_claim_questions(
                        input_text,
                        examples,
                        generated_question=True
                    )
                elif model_args.use_annotated_questions:
                    logging.info("Using claim and QA pairs concat")
                    input_text = concat_claim_questions(
                        input_text,
                        examples,
                        generated_question=False
                    )

        # Tokenize the texts
        result1 = tokenizer(
            input_text1,
            padding="max_length",
            truncation=True,
            max_length=data_args.max_seq_length
        )
        # print(result1)
        # print(type(result1))
        # exit(0)
        result2 = tokenizer(
            input_text2,
            padding="max_length",
            truncation=True,
            max_length=data_args.max_seq_length
        )
        result3 = tokenizer(
            input_text3,
            padding="max_length",
            truncation=True,
            max_length=data_args.max_seq_length
        )
        result = {
            'input_ids': result1['input_ids'],  
            'attention_mask': result1['attention_mask'], 
            'input_ids2': result2['input_ids'],  
            'attention_mask2': result2['attention_mask'], 
            'input_ids3': result3['input_ids'],  
            'attention_mask3': result3['attention_mask'], 
        }
        


        # Map labels to IDs (not necessary for GLUE tasks)
        if label_to_id is not None and "label" in examples:
            result["label"] = [(label_to_id[l] if l != -1 else -1) for l in
                               examples["label"]]
        return result
    with training_args.main_process_first(desc="dataset map pre-processing"):
        raw_datasets = raw_datasets.map(
            convert_raw_to_indices,
            batched=True,
            load_from_cache_file=not data_args.overwrite_cache,
            desc="Running tokenizer on dataset",
        )

    if training_args.do_train:
        if "train" not in raw_datasets:
            raise ValueError("--do_train requires a train dataset")
        train_dataset = raw_datasets["train"]
        if data_args.max_train_samples is not None:
            max_train_samples = min(len(train_dataset), data_args.max_train_samples)
            train_dataset = train_dataset.select(range(max_train_samples))
    else:
        train_dataset = None

    if training_args.do_eval:
        if "validation" not in raw_datasets and "validation_matched" not in raw_datasets:
            raise ValueError("--do_eval requires a validation dataset")
        eval_dataset = raw_datasets["validation"]
        if data_args.max_eval_samples is not None:
            max_eval_samples = min(len(eval_dataset), data_args.max_eval_samples)
            eval_dataset = eval_dataset.select(range(max_eval_samples))
    else:
        eval_dataset = None

    if training_args.do_predict:
        if "test" not in raw_datasets:
            raise ValueError("--do_predict requires a test dataset")
        predict_dataset = raw_datasets["test"]
        if data_args.max_predict_samples is not None:
            max_predict_samples = min(len(predict_dataset), data_args.max_predict_samples)
            predict_dataset = predict_dataset.select(range(max_predict_samples))
    else:
        predict_dataset = None

    es_callback = EarlyStoppingCallback(early_stopping_patience=4)
    # Initialize our Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer,
        callbacks=[es_callback]
    )

    # Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        metrics = train_result.metrics
        max_train_samples = (
            data_args.max_train_samples if data_args.max_train_samples is not None else len(
                train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))

        trainer.save_model()  # Saves the tokenizer too for easy upload

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # Evaluation
    # if training_args.do_eval:
    #     logger.info("*** Evaluate ***")
    #     metrics = trainer.evaluate(eval_dataset=eval_dataset)
    #     max_eval_samples = (
    #         data_args.max_eval_samples if data_args.max_eval_samples is not None else len(
    #             eval_dataset)
    #     )
    #     metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))
    #     trainer.log_metrics("eval", metrics)
    #     trainer.save_metrics("eval", metrics)

    # Prediction
    if training_args.do_predict:
        logger.info("*** Predict ***")
        # metrics = trainer.predict(predict_dataset, metric_key_prefix="test")
        import time
        start_time = time.time()
        prediction_bundle = trainer.predict(predict_dataset, metric_key_prefix="test")
        end_time = time.time()

        inference_time = end_time - start_time
        print(f"inference_time: {inference_time * 1000:.2f} ms")
        exit(0)
        metrics = prediction_bundle.metrics
        predictions = prediction_bundle.predictions
        labels = prediction_bundle.label_ids
        output = []
        for p, l in zip(predictions, labels):
            print(np.argmax(p), l)
            output.append({
                "prediction": int(np.argmax(p)),
                "label": int(l)
            })
        print(output)
        output_predict_file = os.path.join(training_args.output_dir, f"predict_results_{seed}.json")
        max_eval_samples = (
            data_args.max_eval_samples if data_args.max_eval_samples is not None else len(
                eval_dataset)
        )
        json.dump(output, open(output_predict_file, "w"), indent=4)
        metrics["test_samples"] = min(max_eval_samples, len(eval_dataset))
        trainer.log_metrics("test", metrics)
        trainer.save_metrics("test", metrics)


if __name__ == "__main__":
    main()
