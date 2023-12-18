# Use Deepspeed to run Llama-2 on GKE User Guide
In this repo, we will guide you to use deepspeed to train Llama-2 model on GKE step by step. Below are detail steps,

## Clone this repo to your local
```
## clone this repo to your local, then
cd LLM-Tuning-On-GCP/Train/Deepspeed-On-GKE/Llama2-Deepspeed
```

## Set environment variables
```
PROJECT_ID=$(gcloud config get project) e.g., flius-vpc-2
REGION=<GCP region> e.g., us-central1
AR_REPO=<artifacts registry repo name> e.g., flius-vpc-2-repo
BUCKET_NAME=<your GCS bucket name to place model and dataset> e.g., flius-vpc-2-llama-bucket
MODEL_PATH=<model path of GCS bucket subfolder> e.g., llama-2-7b-hf/basemodel
```

## Generate ssh private and public key
```
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
## Download Llama-2-7b-hf model and dataset samsum from huggingface and upload them your your GCS bucket
```
# download llama2 model
git lfs install
git clone https://huggingface.co/meta-llama/Llama-2-7b-hf

# download dataset
https://huggingface.co/datasets/samsum
```
## Use cloud build to build and push docker image to artifacts registry.
```
gcloud builds submit --tag us-docker.pkg.dev/${PROJECT_ID}/${AR_REPO}/llama2:deepspeed
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

## Use workload identity, bind kubernetes service account with gcp service account
```
KSA_NAME=<kubernetes service account> e.g., default
NAMESPACE=<kubernetes namespace> e.g., default
GSA_NAME=<GCP service account for workload identity and to access GCS storage> e.g., fuse-gsa

kubectl create serviceaccount ${KSA_NAME} --namespace ${NAMESPACE}
gcloud iam service-accounts create ${GSA_NAME} --project=${PROJECT_ID}
gcloud storage buckets add-iam-policy-binding gs://${BUCKET_NAME} \
    --member "serviceAccount:${GSA_NAME}@${PROJECT_ID}.iam.gserviceaccount.com" \
    --role "roles/storage.objectAdmin"
gcloud iam service-accounts add-iam-policy-binding ${GSA_NAME}@${PROJECT_ID}.iam.gserviceaccount.com \
    --role roles/iam.workloadIdentityUser \
    --member "serviceAccount:${PROJECT_ID}.svc.id.goog[${NAMESPACE}/${KSA_NAME}]"
kubectl annotate serviceaccount ${KSA_NAME} --namespace ${NAMESPACE} iam.gke.io/gcp-service-account=${GSA_NAME}@${PROJECT_ID}.iam.gserviceaccount.com
```

## Apply kubernetes manifests
```
cp statefulset-llama2-deepspeed-template.yaml statefulset-llama2-deepspeed.yaml
sed -i -e "s@PROJECT_ID@${PROJECT_ID}@g" statefulset-llama2-deepspeed.yaml
sed -i -e "s@AR_REPO@${AR_REPO}@g" statefulset-llama2-deepspeed.yaml
sed -i -e "s@BUCKET_NAME@${BUCKET_NAME}@g" statefulset-llama2-deepspeed.yaml

kubectl apply -f statefulset-llama2-deepspeed.yaml
```

## Run deepspeed inside a pod
```
deepspeed --hostfile=/config/hostfile main.py \
   --sft_only_data_path "samsum" \
   --model_name_or_path "/gcs/${MODEL_PATH}" \
   --per_device_train_batch_size 4 \
   --per_device_eval_batch_size 4 \
   --max_seq_len 512 \
   --learning_rate 9.65e-6 \
   --weight_decay 0. \
   --num_train_epochs 1  \
   --gradient_accumulation_steps 1 \
   --lr_scheduler_type cosine \
   --num_warmup_steps 0 \
   --seed 1234 \
   --gradient_checkpointing \
   --zero_stage 3 \
   --deepspeed \
   --lora_dim 128 \
   --lora_module_name "layers." \
   --only_optimize_lora \
   --print_loss
```
