# AAU-THESIS
**Title:** Using Large Language Models for Aspect Based Sentiment Analysis

**Student:** Sadiksha Baral

**Supervisor:** Hamid Bekamiri


## Repository Structure

This repository contains the code used for the AAU thesis. Below is the detailed structure of the repository:

### 0_data
This directory contains the data processing and preparation scripts.

- **data_processing.ipynb**: Jupyter notebook for initial data processing.
- **df_to_dataset.ipynb**: Notebook to convert DataFrame to HF dataset format.
- **openai_data_prep_finetune.ipynb**: Preparation of data for fine-tuning OpenAI models.

### 1_simple_models
This directory includes implementations of simple machine learning models.

- **02_SVM_boosting.ipynb**: Jupyter notebook for Support Vector Machine (SVM) and gradient boosting.
- **03_LSTM.ipynb**: Implementation of Long Short-Term Memory (LSTM) network.
- **04_BERT.ipynb**: Fine-tuning and usage of BERT model.

### 2_llms
This directory focuses on large language models.

- **setfitforABSA.ipynb**: Setup and fine-tuning for Aspect-Based Sentiment Analysis (ABSA) with Setfit (sbert, sentence-transformers).
- **AWQ_mistral.ipynb**: Script related to the AWQ Mistral model.

- **phi3_prompting.ipynb**: Prompting techniques and experiments for Phi3 model.
- **phi3.ipynb**: General setup and N shot classfication with phi3.
- **openai.ipynb**: OpenAI GPT3.5 turbo finetuning and inference notebook.
- **openai.jsonl**: Sampled data for gpt3.5 turbo finetuning in json lines format.


### 3_eval
This directory contains evaluation scripts and results.

- **results**: Folder containing evaluation results (csv files).
- **evaluation_openai.ipynb**: Evaluation notebook for OpenAI models.
- **evaluation_phi3_nshot.ipynb**: Few-shot evaluation of phi3 model.
- **evaluation_phi3_prompting.ipynb**: Evaluation of prompting techniques for phi3 model.

### Other Files
- **requirements.txt**: List of general dependencies (Some notebooks require specific versions of packages, so are reinstalled on notebook themselves)

## Prerequisites
Ensure you have the following installed:
- Python 3.8 or higher
- Dependencies listed in `requirements.txt`

## Getting Raw data
Raw data is not uploaded here on Github but can be accessed from the following GOogle drive link.

[AAU thesis data - Google Drive](https://drive.google.com/drive/folders/1VeH-2A_aBUzQrlYp7ugor0giYAPLGfHt?usp=sharing)