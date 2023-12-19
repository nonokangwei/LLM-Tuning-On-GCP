# LLM-Tuning-On-GCP: Serve
This folder demonstrates how to serve Llama2 model and other LLMs on Google Cloud.


## Framework
At the Serve part, we will use two Frameworks. vLLM as the backend serve server for its high througput and low latency, FastChat as the front Chat web Server for its easy to interact webui. You can get more information and code in Serve/corresponding_folder.

- vLLM: [vLLM](https://github.com/vllm-project/vllm)
- FastChat: [FastChat](https://github.com/lm-sys/FastChat)


## Supported Models
The Serve solution support pretrained model, full parameter fine tuned model, Adapter. You need to download the model file on **GCS** before serving.

### Pretrained model, Full parameter fine tuned model
In this guide, we will use Llama2 model. vLLM and FastChat seamlessly supports many models, you can check if your model is supported using the belowing:

- vLLM supported Models: [vLLM supported Models](https://docs.vllm.ai/en/latest/models/supported_models.html)
- FastChat supported Models: [FastChat supported Models](https://github.com/lm-sys/FastChat/blob/main/docs/model_support.md)

### Adapter supported
vLLM and FastChat can not load Fine Tune Adapaters (such as Lora Adapter) automatically. In this solution, we added support for adapter through merge base model and adapter. 


## Cloud Services
In this guide, we will serve Llama2 model on 3 platforms: **GCE**, **GKE**, **Vertex AI**.

All Google Cloud GPUs (H100, A100 80G, A100 40G, V100, L4, T4), are supported in these three platforms

## Serve Steps
If you want to serve LLM using vLLM on GCE, go to [vLLM-on-GCE](./vLLM-on-GCE/) folder.

If you want to serve LLM using vLLM on GKE, go to [vLLM-on-GKE](./vLLM-on-GKE/) folder.

If you want to serve LLM on GKE, using vLLM as backend and FastChat as front web ui, go to [FastChat-with-vLLM-on-GKE](./FastChat-with-vLLM-on-GKE/) folder.

If you want to serve LLM on Vertex AI, go to [vLLM-on-Vertex](./vLLM-on-Vertex/)folder.
