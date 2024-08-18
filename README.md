# Advanced Claim Identification Library

Guneesh Vats | IIIT Hyderabad | guneesh.vats@research.iiit.ac.in | Github Repo Link



## Overview
Welcome to the Text Analysis Library, a comprehensive solution for advanced text analysis tasks! This library leverages state-of-the-art language models and embedding techniques to provide robust functionality for zero-shot classification and fine-tuning of models.

**Key Features:**
* Zero-Shot Classification: Classify text into predefined categories without any prior training, using powerful models like BART and Llama2.
* Fine-Tuning: Customize pre-trained models to better suit your specific dataset, enhancing their performance for your unique requirements.
* Output Generation: Automatically generate detailed output files in both CSV and JSON formats, including record IDs, extracted claims, and accuracy scores.
* Accuracy Metrics: Compute and log accuracy scores to evaluate the performance of your models, ensuring reliable and accurate text analysis.


## Installation
```git clone https://github.com/your-repo/text_analysis_lib.git```

```cd text_analysis_lib```

```pip install -i requirements.txt```

```pip install -e .```


## Usage

### Zero Shot Classification 
    python -m text_analysis_lib.zero_shot_classification --data_path path_to_your_data.csv --output_path path_to_output.csv --token your_huggingface_token

--data_path: Path to the input CSV file

--output_path: Path to the output CSV file

--token: Hugging Face token for authentication

### Fine Tuning 
    python -m text_analysis_lib.fine_tuning --data_path path_to_your_data.csv --model_name model_name_to_fine_tune --token your_huggingface_token --output_model_path path_to_save_fine_tuned_model

--data_path: Path to the input CSV file

--model_name: Model name to fine-tune

--token: Hugging Face token for authentication

--output_model_path: Path to save the fine-tuned model

### Generating CSV of Fine Tuned Model Output
    python -m text_analysis_lib.generate_csv --data_path path_to_your_data.csv --model_path path_to_fine_tuned_model --output_path path_to_output.csv

--data_path: Path to the input CSV file

--model_path: Path to the fine-tuned model

--output_path: Path to the output CSV file

--token: Hugging Face token for authentication

### Optional cli parse argument values for LLM models 

- ```facebook/bart-large```
- ```facebook/bart-large-mnli```
- ```meta-llama/Llama-2-7b-chat-hf```
- ```bert-base-uncased```
- ```roberta-base```
- ```distilbert-base-uncased```
- ```microsoft/deberta-v3-base```


## Files Structure

```text_analysis_lib/
│
├── README.md
├── setup.py
├── text_analysis_lib/
│   ├── __init__.py
│   ├── zero_shot_classification.py
│   ├── fine_tuning.py
│   ├── generate_csv.py
│   └── utils.py
├── report/
│   └── report.md
├── output_files/
│   ├── output.json
│   ├── preprocessed_data_finetuned_bart_FT1.csv
├── optional_files/
│   ├── BART_eval_log_finetuned_final.txt
│   ├── BART_testeval_log.txt
│   ├── BART_ZSCeval_log.txt
│   ├── ZSC_BART_Cosine_bert-base-nli.py
│   ├── ZSC_LLAMA2_bert_base_nli.py
│   ├── Training_BART_bert-base-nli.py
│   └── FT1_csv_maker_BART_bert_base_nli.py```


