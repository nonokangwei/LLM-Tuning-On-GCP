# Use Deepspeed to run T5 on GKE User Guide
In this repo, we will guide you to use deepspeed to train T5 model on GKE step by step. Below are detail steps,

## Clone this repo to your local
```
## clone this repo to your local, then
cd LLM-Tuning-On-GCP/Train/Deepspeed-On-GKE/T5-Deepspeed
```

## Set environment variables
```
PROJECT_ID=$(gcloud config get project) e.g., flius-vpc-2
REGION=<GCP region> e.g., us-central1
AR_REPO=<artifacts registry repo name> e.g., flius-vpc-2-repo
BUCKET_NAME=<your GCS bucket name to place model and dataset> e.g., flius-vpc-2-bucket
MODEL_PATH=<model path of GCS bucket subfolder> e.g., llama-2-7b-hf/basemodel
```

## Generate ssh private and public key
```
ssh-keygen -t rsa -f ./id_rsa
cp id_rsa.pub authorized_keys
```

## Use cloud build to build and push docker image to artifacts registry.
```
gcloud builds submit --tag us-docker.pkg.dev/${PROJECT_ID}/${AR_REPO}/t5:deepspeed
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
cp statefulset-t5-deepspeed-template.yaml statefulset-t5-deepspeed.yaml
sed -i -e "s@PROJECT_ID@${PROJECT_ID}@g" statefulset-t5-deepspeed.yaml
sed -i -e "s@AR_REPO@${AR_REPO}@g" statefulset-t5-deepspeed.yaml
sed -i -e "s@BUCKET_NAME@${BUCKET_NAME}@g" statefulset-t5-deepspeed.yaml

kubectl apply -f statefulset-t5-deepspeed.yaml
```

## Run deepspeed inside a pod
```
nohup deepspeed --hostfile=/config/hostfile run.py --batch_size=8 --epoch=1 --train_dataset_path=/gcs/deepspeed/data/train --test_dataset_path=/gcs/deepspeed/data/eval --model_output_dir=/model --tensorboard_log_dir=/log &
```

## Performance benchmark (GCE, GKE, Vertex)
|| 4 machines 8 GPUs | 2 machines 4 GPUs |
| -------- | ------- | ------- |
| GCE | 1h40min(mtu=8228) |  |
| GKE | 1h20min(mtu=8228) 1h15min(fast socket, gvnic, mtu=8228) | 2h35min|
| Vertex AI | 1h45min (w/o resource provisioning time) | 3h / 3h (fast socket)|
