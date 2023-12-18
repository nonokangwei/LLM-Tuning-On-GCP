# Serve Llama 7b using vLLM on GCE using L4 GPUs

In this guide, we will introduce how to use vLLM serve Llama base model, pretrained model and Lora adapter model on GCE.


## Prerequisites
*   A terminal with `kubectl` and `gcloud` installed. Cloud Shell works great!
*   L4 GPUs quota to be able to run additional 2 L4 GPUs
*   You have access of llama2 model,  or you have compelete Training and had pretrained model or Lora adapter model on GCS bucket

## Create GPU VM instance

```bash
export PROJECT=<project_name>
export REGION=<region>
export ZONE=<vm_instance_zone>
export VPC=<vpc_name>
export SUBNET=<vpc_subnet>
export VM_NAME=<vm_instance_name>
export TYPE=<gpu_machine_type>
export ACCELERATOR_TYPE=<gpu_machine_type>
export ACCELERATOR_NUMBER=<number>
export SA=<vm_service_account>


gcloud compute instances create ${VM_NAME} \
    --project=${PROJECT} \
    --zone=${ZONE} \
    --machine-type=${TYPE} \
    --network-interface=network-tier=PREMIUM,subnet=${SUBNET} \
    --maintenance-policy=TERMINATE \
    --provisioning-model=STANDARD \
    --service-account=${SA} \
    --accelerator=count=${ACCELERATOR_NUMBER},type=${ACCELERATOR_TYPE} \
    --tags=vllm-server-tag \
    --boot-disk-size=${BOOT_DISK_SZ:-100} \
    --boot-disk-type=pd-balanced

gcloud compute firewall-rules create allow-vllm-server \
   --project=${PROJECT} \
   --direction=INGRESS \
   --priority=1000 \
   --network=${VPC} \
   --action=ALLOW --rules=tcp:8000 \
   --source-ranges=0.0.0.0/0 \
   --target-tags=vllm-server-tag

```



## SSH into VM and Install GPU driver
```bash
#ssh
gcloud compute ssh ${VM_NAME?} --zone=${ZONE?} --project=${PROJECT?}


# gpu driver
curl https://raw.githubusercontent.com/GoogleCloudPlatform/compute-gpu-installation/main/linux/install_gpu_driver.py --output install_gpu_driver.py

sudo python3 install_gpu_driver.py

sudo nvidia-smi

```

## Install vLLM

### Install vLLM using pip
```bash
sudo apt-get update
sudo apt-get install git
sudo apt-get install pip

pip install vllm
pip install peft
pip install ray
pip install aiohttp
```

### Download model file
In this guide, we will use vLLM to serve three models: Llama base model, Lora fine tune model, pretrained model

You can download models from gcs

```bash
mkdir models
mkdir models/base_model
mkdir models/lora_model
mkdir models/pretrained_model

cd models/base_model
gcloud storage cp gs://your_base_model_path

cd models/lora_model
gcloud storage cp gs://your_lora_model_path

cd models/pretrained_model
gcloud storage cp gs://your_pretrained_model_path

```

You can also download models from hugging face
```bash
cd models

sudo apt-get install git-lfs
git lfs install
git clone https://huggingface.co/meta-llama/Llama-2-7b-chat-hf
```


### (optional) Merge base model and lora adapter


Vllm does not support the Lora adapter now. So if you want to serve a Lora fine tune model, you should merge the base model and the Lora adapter first.

In this guide, we used peft merge_and_unload to achieve this.

vim merge_peft.py

[embedmd]:# (merge_peft.py)
```python
# Example usage:
# python3 merge_peft.py --base_model=models/base_model --peft_model=models/lora_model --saved_path=models/merged_model


from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch

import argparse

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", type=str)
    parser.add_argument("--peft_model", type=str)
    parser.add_argument("--saved_path", type=str)

    return parser.parse_args()

def main():
    args = get_args()

    print(f"[1/4] Loading base model: {args.base_model}")
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        return_dict=True,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)

    print(f"[2/4] Loading adapter: {args.peft_model}")
    model = PeftModel.from_pretrained(base_model, args.peft_model, device_map="auto")
    
    print("[3/4] Merge base model and adapter")
    model = model.merge_and_unload()
    model.save_pretrained(args.saved_path)
    tokenizer.save_pretrained(args.saved_path)
    
    print("[4/4] Save merged model to local path")

if __name__ == "__main__" :
    main()

```

Run merge_peft.py
```bash
python3 merge_peft.py --base_model=models/base_model --peft_model=models/lora_model --saved_path=models/merged_model
```

## Deploy as an api server

You can use tensor-parallel-size to specify the number of gpu to run multi-GPU serving.
```bash
# serve llama2 7b origin model
python3 -O -u -m vllm.entrypoints.api_server\
    --host=0.0.0.0 \
    --port=8000 \
    --model=models/llama-2-7b-chat-hf \
    --tensor-parallel-size 2


# serve lora adapter fine tune llama2 7b merged model
python3 -O -u -m vllm.entrypoints.api_server\
    --host=0.0.0.0 \
    --port=8000 \
    --model=models/merged_model \
    --tensor-parallel-size 2 
```

Then you can send post request to query the model
```bash
curl http://vm_public_ip:8000/generate \
    -d '{
        "prompt": "San Francisco is a",
        "use_beam_search": true,
        "n": 4,
        "temperature": 0
    }'

```

## Benchmark
Downloading the ShareGPT dataset
```bash
mkdir datasets
cd datasets

wget https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json
```

Run benchmark test using vllm test script


```bash
git clone https://github.com/vllm-project/vllm.git

python3 vllm/benchmarks/benchmark_serving.py --backend vllm --tokenizer models/llama-2-7b-chat-hf --dataset datasets/ShareGPT_V3_unfiltered_cleaned_split.json --host 0.0.0.0 --port 8000
```
