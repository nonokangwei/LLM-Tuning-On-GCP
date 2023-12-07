# LLM-Tuning-On-GCP

This folder demonstrates how to tune Llama2 model on Google Cloud.

## Framework
Tuning is based on two frameworks. You can get the code in training/corresponding_folder
1. Deepspeed: [Deepspeed-Chat](https://github.com/microsoft/DeepSpeedExamples/tree/master/applications/DeepSpeed-Chat)
2. Pytorch FSDP: [Llama-Recipes](https://github.com/facebookresearch/llama-recipes/tree/main)

## Cloud Services
Tuning is on **Vertex AI** and **GKE**. 

## Tuning Steps
If you want to run tuning on Vertex AI, go to [Train](./Train) folder, and select [Deepspeed](./Train/Deepspeed) or [FSDP](./Train/FSDP) as framework. 

For FSDP, just step by step run the [quick_start](./Train/FSDP/quick_start.ipynb) notebook. 

For Deepspeed, there's two implementations. One is using [Deepspeed launcher](./Train/Deepspeed/deepspeed-launcher/) to launch distributed training job. It is referred from [this repo](https://github.com/gkcng/vertex-deepspeed/tree/main). The other is using [Torchrun launcher](./Train/Deepspeed/torchrun-launcher/) to launch distributed training job. Just step by step run Vertex_DeepspeedChat.ipynb notebook in each folder.

If you want to run tuning on GKE, go to [Train](./Train) folder, and go to [Deepspeed-On-GKE](./Train/Deepspeed-On-GKE/). There are several samples. including [T5](./Train/Deepspeed-On-GKE/T5-Training/), [Deepspeed-Chat](./Train/Deepspeed-On-GKE/DeepSpeed-Chat/), [HelloDeepspeed](./Train/Deepspeed-On-GKE/HelloDeepSpeed/) from Deepspeed public repo, and [Torchrun](./Train/Deepspeed-On-GKE/Torchrun/) version of Deepspeed-Chat.




