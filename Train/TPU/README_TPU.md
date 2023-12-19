# LLM-Tuning-On-GCP-TPU

This document describes how to pre-train/fine-tune Llama2 model on Google Cloud TPU with our demo code.

## Prerequisits
Contact your GCP representative to grant access to this repository [cloud-ce-share-csr/wwoo-llama-xla-fsdp](https://source.cloud.google.com/cloud-ce-shared-csr/wwoo-llama-xla-fsdp/) \
Setup [Google Cloud SDK](https://cloud.google.com/sdk/docs/install) \
Test if you have access to clone this repository with below commands
```
gcloud init
gcloud source repos clone wwoo-llama-xla-fsdp --project=cloud-ce-shared-csr
```

## Create Cloud TPU

```
export PROJECT=<project_name>
export REGION=<region>
export ZONE=<tpu_vm_instance_zone>
export VPC=<vpc_name>
export SUBNET=<vpc_subnet>
export TPUVM=<tpu_vm_instance_name>
export TYPE=<tpu_vm_type> # v3-8 at minimum for Llama-7B, recommand v3-32 or more.
export SA=<tpu_vm_service_account>
export IMAGE=<tpu_vm_image> # recommend tpu-ubuntu2204-base 

gcloud compute tpus tpu-vm create ${TPUVM} \
	--zone=${ZONE} \
	--accelerator-type=${TYPE} \
	--version=${IMAGE} \
	--network=${VPC} \
	--subnetwork="projects/${PROJECT}/regions/${REGION}/subnetworks/${SUBNET}" \
	--internal-ips \
	--service-account=${SA}

```

## Setup environments and clone repository
*NOTE*: If you are using a single worker(e.g. v3-8 which has 1 node only), you can directly ssh login like
```
gcloud alpha compute tpus tpu-vm ssh ${TPUVM} --zone=${ZONE} --tunnel-through-iap
```
otherwise, run commands on all workers like below
```
gcloud alpha compute tpus tpu-vm ssh ${TPUVM} --zone ${ZONE} --worker all --tunnel-through-iap --command=''
```
### Update with latest torch/XLA
```
gcloud alpha compute tpus tpu-vm ssh ${TPUVM} --zone ${ZONE} --worker all --tunnel-through-iap --command='
export NEEDRESTART_MODE=a
export DEBIAN_FRONTEND=noninteractive
sudo apt-get remove needrestart -y
sudo apt update -y && sudo apt upgrade -y
pip install torch~=2.1.0 torch_xla[tpu]~=2.1.0 -f https://storage.googleapis.com/libtpu-releases/index.html
pip install datasets evaluate scikit-learn accelerate py7zr
'
```

### Setup gcsfuse(optional)
TPU VM with multiple workers(e.g. v3-32 has 4 nodes, check it with
``` 
gcloud compute tpus describe ${TPUVM}
```
) need a shared filesystem for storing sharing files, gcsfuse is one of the best options for this case.

For more about TPU storage options,\
https://cloud.google.com/tpu/docs/storage-options

For service account setting about TPU gcsfuse access,\
https://cloud.google.com/tpu/docs/storage-buckets

For more mount options,\
https://cloud.google.com/storage/docs/gcsfuse-mount#mount-bucket

```
# Setup service account used by TPU
gcloud beta services identity create --service tpu.googleapis.com --project ${PROJECT}

# Create buckets and grant service account with relevant ACL
gsutil mb -l ${REGION} gs://${BUCKET}/
gsutil acl ch -u ${TPUVM_SERVICE_ACCOUNT}:READER gs://${BUCKET}
gsutil acl ch -u ${TPUVM_SERVICE_ACCOUNT}:WRITER gs://${BUCKET}

# Install gcsfuse and mount
gcloud alpha compute tpus tpu-vm ssh ${TPUVM} --zone ${ZONE} --worker all --tunnel-through-iap --command='
export GCSFUSE_REPO=gcsfuse-$(lsb_release -c -s)
export NEEDRESTART_MODE=a
export DEBIAN_FRONTEND=noninteractive
echo "deb https://packages.cloud.google.com/apt $GCSFUSE_REPO main" | sudo tee /etc/apt/sources.list.d/gcsfuse.list
curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo apt-key add -
sudo apt-get update -y
sudo apt-get install fuse gcsfuse -y
gcsfuse -v

sudo mkdir -p /mnt/data/
sudo mount -t gcsfuse -o rw,allow_other,uid=${USER},gid=${USER} ${BUCKET} /mnt/data/
'
```

### Clone repository and install requirements
```
gcloud init
gcloud source repos clone wwoo-llama-xla-fsdp --project=cloud-ce-shared-csr
cd wwoo-llama-xla-fsdp
python3 setup.py sdist --formats=gztar

# Copy to all TPU workers
gcloud alpha compute tpus tpu-vm scp dist/llama-xla-0.1.tar.gz ${TPUVM}: --tunnel-through-iap --zone ${ZONE}

# Install llama-xla
gcloud alpha compute tpus tpu-vm ssh ${TPUVM} --zone ${ZONE} --worker all --tunnel-through-iap --command='
pip install llama-xla-0.1.tar.gz
'
```

### Download Llama2 model folder
In this example, I used the downloaded Llama2. You can use huggingface’s model name as parameters instead after you have
```
huggingface-cli login --token <your_hf_token>
```
with access to https://huggingface.co/meta-llama/Llama-2-7b-hf.

If what you have is the RAW model from Meta, you can follow below instruction to convert to huggingface’s format.
https://github.com/facebookresearch/llama-recipes?tab=readme-ov-file#model-conversion-to-hugging-face

## Training on TPU
### Run the pre-training job
*Note*: TPU v3-8 has 4 chips, each chip comes with 2 cores. So num_cores=4 for v3-8 while num_cores=16 for v3-32.
For more details, please refer to https://github.com/pytorch/xla/blob/master/docs/pjrt.md#multithreading-on-tpu-v2v3
```
gcloud alpha compute tpus tpu-vm ssh ${TPUVM} --zone ${ZONE} --worker all --tunnel-through-iap --command='
export PJRT_DEVICE=TPU
export PT_XLA_DEBUG=1
export USE_TORCH=ON
export XLA_USE_BF16=1

python3 -m llama_xla.train \
	--num_cores 16 \
	--config /mnt/data/llama-2-7b-hf/config.json \
	--expanded_checkpoint_dir /mnt/data/hf-7B-expanded \
	--expand_checkpoint_on_master_only \
	--tokenizer /mnt/data/llama-2-7b-hf/ \
	--dataset_name samsum \
    --train_split train \
	--num_epochs 2 \
	--report_steps 100 \
	--block_size 2048 \
	--logging_steps 100 \
	--train_batch_size 2 \
	--save_strategy none \
	--enable_checkpoint_consolidation \
	--enable_gradient_checkpointing \
	--output_dir /mnt/data/llama-2-7b-hf-train-xla/
'
```

### Or run the fine-tuning job
```
gcloud alpha compute tpus tpu-vm ssh ${TPUVM} --zone ${ZONE} --worker all --tunnel-through-iap --command='
export PJRT_DEVICE=TPU
export PT_XLA_DEBUG=1
export USE_TORCH=ON
export XLA_USE_BF16=1

python3 -m llama_xla.train \
	--num_cores 16 \
	--huggingface_model_dir /mnt/data/llama-2-7b-hf/ \
	--config /mnt/data/llama-2-7b-hf/config.json \
	--expanded_checkpoint_dir /mnt/data/hf-7B-expanded \
	--expand_checkpoint_on_master_only \
	--tokenizer /mnt/data/llama-2-7b-hf/ \
	--dataset_name samsum \
    --train_split train \
	--num_epochs 2 \
	--report_steps 100 \
	--block_size 2048 \
	--logging_steps 100 \
	--train_batch_size 2 \
	--save_strategy none \
	--enable_checkpoint_consolidation \
	--enable_gradient_checkpointing \
	--output_dir /mnt/data/llama-2-7b-hf-ft-xla/
'
```
### Checkpoint consolidation
Since above command already included `--enable_checkpoint_consolidation` flag, you should be able to get a consolidated model file to be used.
Otherwise, use below utility to consolidate the sharded model files.
```
python3 -m llama_xla.utils.weights \
   --function convert_sharded_checkpoint_to_hf
   --model_dir . \
   --output_dir .
```
PyTorch XLA provides an "out-of-box" torch_xla.distributed.fsdp.consolidate_sharded_ckpts command line utility. We do not use this here because it nests the model state (parameters) under a "model" state key, which HuggingFace transformers does not expect.



