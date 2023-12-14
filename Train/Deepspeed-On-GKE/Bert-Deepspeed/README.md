# Use Deepspeed to run T5 on GKE User Guide
In this repo, we will guide you to use deepspeed to train T5 model on GKE step by step. Below are detail steps,

## Clone this repo to your local
```
## clone this repo to your local, then
cd LLM-Tuning-On-GCP/Train/Deepspeed-On-GKE/Bert-Deepspeed
```

## Set environment variables
```
PROJECT_ID=$(gcloud config get project) e.g., flius-vpc-2
REGION=<GCP region> e.g., us-central1
AR_REPO=<artifacts registry repo name> e.g., flius-vpc-2-repo
```

## Generate ssh private and public key
```
ssh-keygen -t rsa -f ./id_rsa
cp id_rsa.pub authorized_keys
```

## Use cloud build to build and push docker image to artifacts registry.
```
cp Dockerfile.ubuntu18.04.base Dockerfile
gcloud builds submit --tag us-docker.pkg.dev/${PROJECT_ID}/${AR_REPO}/bert:deepspeed-base
cp Dockerfile.ubuntu18.04.hellodeepspeed Dockerfile
sed -i -e "s@PROJECT_ID@${PROJECT_ID}@g" Dockerfile
sed -i -e "s@AR_REPO@${AR_REPO}@g" Dockerfile
gcloud builds submit --tag us-docker.pkg.dev/${PROJECT_ID}/${AR_REPO}/bert:deepspeed
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
  --num-nodes=0 --min-nodes=0 --max-nodes=4 \
  --shielded-secure-boot \
  --shielded-integrity-monitoring \
  --node-locations $REGION-a,$REGION-b --region $REGION --spot
```
## Apply kubernetes manifests
```
cp statefulset-bert-deepspeed-template.yaml statefulset-bert-deepspeed.yaml
sed -i -e "s@PROJECT_ID@${PROJECT_ID}@g" statefulset-bert-deepspeed.yaml
sed -i -e "s@AR_REPO@${AR_REPO}@g" statefulset-bert-deepspeed.yaml

kubectl apply -f statefulset-bert-deepspeed.yaml
```

## Run deepspeed inside a pod
```

cd /tmp/DeepSpeedExamples/training/HelloDeepSpeed
deepspeed  --hostfile=/config/hostfile train_bert.py --checkpoint_dir .
```
