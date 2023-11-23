# Use Torchrun to run Llama-2 on GKE User Guide
## Clone this repo to your local
```bash
git clone https://github.com/Leisureroad/LLM-Tuning-On-GCP.git
cd LLM-Tuning-On-GCP/Train/Deepspeed-On-GKE/Torchrun
```

## Generate ssh private and public key
```bash
ssh-keygen -t rsa -f ./id_rsa
cp id_rsa.pub authorized_keys
```

## Access to llama2 model
We will use huggingface to do training, so you need to request the access to download llama2 model in huggingface.
https://huggingface.co/meta-llama/Llama-2-7b-chat-hf/tree/main

Create an access token for huggingface (https://huggingface.co/settings/tokens), and create a local token file.
```bash
touch token
```

## Use cloud build to build and push docker image to artifacts registry.
```bash
PROJECT_ID=<your project id>
AR_REPO=<artifacts registry repo name>
gcloud builds submit --tag us-docker.pkg.dev/${PROJECT_ID}/${AR_REPO}/deepspeed-chat:torchrun
```

## Choose your region and set your project:
```bash
export REGION=us-central1
export PROJECT_ID=$(gcloud config get project)
```
## Create GKE cluster and nodepool
Create a GKE cluster:
```bash
gcloud container clusters create l4-demo --location ${REGION} \
  --workload-pool ${PROJECT_ID}.svc.id.goog \
  --enable-image-streaming --enable-shielded-nodes \
  --shielded-secure-boot --shielded-integrity-monitoring \
  --enable-ip-alias \
  --node-locations=$REGION-a \
  --workload-pool=${PROJECT_ID}.svc.id.goog \
  --addons GcsFuseCsiDriver   \
  --no-enable-master-authorized-networks \
  --machine-type n2d-standard-4 \
  --num-nodes 1 --min-nodes 1 --max-nodes 5 \
  --ephemeral-storage-local-ssd=count=2 \
  --enable-ip-alias \
  --enable-private-nodes  \
  --master-ipv4-cidr 172.16.0.32/28
```

Create a nodepool where each VM has 2 x L4 GPU:
```bash
gcloud container node-pools create g2-standard-24 --cluster l4-demo \
  --accelerator type=nvidia-l4,count=2,gpu-driver-version=latest \
  --machine-type g2-standard-24 \
  --ephemeral-storage-local-ssd=count=2 \
  --enable-autoscaling --enable-image-streaming \
  --num-nodes=0 --min-nodes=0 --max-nodes=3 \
  --shielded-secure-boot \
  --shielded-integrity-monitoring \
  --node-locations $REGION-a,$REGION-b --region $REGION --spot
```

## Apply kubernetes manifests
```bash
sed -i -e "s@PROJECT_ID@${PROJECT_ID}@g" job-deepspeed-torchrun.yaml
sed -i -e "s@AR_REPO@${AR_REPO}@g" job-deepspeed-torchrun.yaml

kubectl create -f job-deepspeed-torchrun.yaml
```
