## Requirements
To install requirements:

```setup
conda create -n mocheg python=3.8.10
conda install pytorch==1.9.0 torchvision==0.10.0 torchaudio==0.9.0 cudatoolkit=10.2 -c pytorch
pip install -r requirements.txt
```

## Dataset - MOCHEG

You can download dataset here:

- [MOCHEG version 1](https://docs.google.com/forms/d/e/1FAIpQLScAGehM6X9ARZWW3Fgt7fWMhc_Cec6iiAAN4Rn1BHAk6KOfbw/viewform?usp=sf_link). 
- Place the downloaded dataset in the data/ directory

- Dataset Format and structure are explained in document/MOCHEG_dataset_statement.pdf.

## Pre-trained Models

Download pretrained models to get started:

- [pretrained models](http://nlplab1.cs.vt.edu/~menglong/project/multimodal/fact_checking/MOCHEG/checkpoint) trained on MOCHEG.
- Place the downloaded models in the checkpoint/ directory

## Evidence Retrieval

To retrieve evidence for claims, run:

```eval
python retrieval/utils/preprocess.py
python retrieve_train.py --mode=test --train_config=IMAGE_MODEL --model_name=checkpoint/image_retrieval
python retrieve_train.py --mode=test --train_config=BI_ENCODER --model_name=checkpoint/text_retrieval/bi_encoder
```
 
-----------------------------------------------------------------------------------------

## Counterfactual Instance Construction
Construct counterfactual instances using gold and retrieved evidence:
```eval
python main.py --mode=preprocess_for_verification
```

## Training
Train with gold evidence:
```train
python verify.py --mode=train \
    --model_type=CLAIM_TEXT_IMAGE_attention_5_4 \
    --batch_size=2048 \
    --lr=0.00001 \
    --loss_weight_power=3 \
    --temperature=1
```
## Evaluation
Evaluate with gold evidence:
```train
python verify.py --mode=test \
    --model_type=CLAIM_TEXT_IMAGE_attention_5_4 \
    --batch_size=2048 \
    --lr=0.00001 \
    --loss_weight_power=3 \
    --temperature=1 \
    --checkpoint_dir=xxx
```
Evaluate with system evidence:
```train
python verify.py --mode=test \
    --model_type=CLAIM_TEXT_IMAGE_attention_5_4 \
    --batch_size=2048 \
    --lr=0.00001 \
    --loss_weight_power=3 \
    --temperature=1 \
    --checkpoint_dir=xxx \
    --evidence_file_name=retrieval/retrieval_result.csv
```

## Explanation Generation (Optional)
```
python main.py --mode=preprocess_for_generation_inference
```
```
CUDA_VISIBLE_DEVICES=0 python controllable_generation.py \
    --do_predict \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 8  \
    --predict_with_generate \
    --text_column truth_claim_evidence \
    --summary_column cleaned_ruling_outline  \
    --model_name_or_path #(your_path_to_generation_model, e.g., checkpoint/controllable_generation/generation/without)  \
    --test_file #(your_path_to_preprocessed_data, e.g., data/test/Corpus2_for_controllable_generation.csv)  \
    --train_file #(your_path_to_preprocessed_data, e.g., data/train/Corpus2_for_controllable_generation.csv)   \
    --validation_file #(your_path_to_preprocessed_data, e.g., data/val/Corpus2_for_controllable_generation.csv)   \
    --output_dir controllable/generation/output/bart/run_0/without \--classifier_checkpoint_dir=#(your_path_to_explanation_classify_model, e.g., checkpoint/controllable_generation/explanation_classify)
```

## Acknowledgments
Our repository builds on [Mocheg](https://github.com/PLUM-Lab/Mocheg). Thanks for open-sourcing!


