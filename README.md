LLM tuning on GCP is a repo library for training, serving and MLOps LLM model on Google Cloud Platform.

## Why does this project exist?
LLM has shown state of the art performance across domain, tasks. LLM adoption has rapid growth, foundation model is never the gold belt for domain specific task or vertical use case. Build/Tuning 1st party model will help customer to improve the accuracy and efficiency for the domain specfic task or vertical use case. 

With this repo, customers can leverage GCP engineer pre-proofed code and script to start the journey of LLM training/tuning, serving, and MLOps, We hope this guide will be the lighthouse on the GCP LLM journey.

## Major Features
- Training: [Training/Finetuning with Deepspeed on Vertex AI](./Train/README.md), [Training/Finetuning with FSDP on Vertex AI](./Train/FSDP-on-Vertex/quick_start.ipynb), [Training/Finetuning with Deepspeed on GKE](./Train/Deepspeed-On-GKE/README.md), [Training/Finetuning on TPU](./Train/TPU/README_TPU.md)
- Serving: [Serving with vLLM on GCE](./Serve/vLLM-on-GCE/README.md), [Serving with vLLM on GKE](./Serve/vLLM-on-GKE/README.md), [Serving with vLLM on Vertex AI](./Serve/vLLM-on-Vertex/serving_quick_start.ipynb), [Serving with FastChat on GKE](./Serve/FastChat-with-vLLM-on-GKE/README.md)
- MLOps: [End to End MLOps LLM pipeline (from training to serving) on Vertex AI](./MLOps/README.md)

## Usage
Detailed documentation can be in the subfolder(Train, Serve, MLOps) 