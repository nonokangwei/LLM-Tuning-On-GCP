# LLM-Tuning-On-GCP

This repo demonstrates how to tune and serve Llama2 model on Google Cloud.

## Framework
Tuning is based on two frameworks. You can get the code in training/corresponding_folder
1. Deepspeed: [Deepspeed-Chat](https://github.com/microsoft/DeepSpeedExamples/tree/master/applications/DeepSpeed-Chat)
2. Pytorch FSDP: [Llama-Recipes](https://github.com/facebookresearch/llama-recipes/tree/main)

Serving is based on two frameworks.
1. vLLM: [vLLM](https://github.com/vllm-project/vllm)
2. TensorRT: [TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM/tree/release/0.5.0)

## Cloud Services
Tuning and serving are both on **Vertex AI**. Vertex AI Pipeline is used to orchestrate the training (Vertex custom training) and serving (Vertex Endpoint) workflow. Deepspeed-Chat on Vertex AI is referred from [this repo](https://github.com/gkcng/vertex-deepspeed/tree/main).

## Tuning Steps
If you want to run tuning on Vertex AI, go to [Train](./Train) folder, and select [Deepspeed](./Train/Deepspeed) or [FSDP](./Train/FSDP) as framework. 

For FSDP, just step by step run the [quick_start](./Train/FSDP/quick_start.ipynb) notebook. 

For Deepspeed, there's two implementations. One is using [Deepspeed launcher](./Train/Deepspeed/deepspeed-launcher/) to launch distributed training job. The other is using [Torchrun launcher](./Train/Deepspeed/torchrun-launcher/) to launch distributed training job. Just step by step run Vertex_DeepspeedChat.ipynb notebook in each folder.

## TBC




