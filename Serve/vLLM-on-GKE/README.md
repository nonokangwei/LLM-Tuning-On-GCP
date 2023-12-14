# Serve Llama 7b using vLLM on GKE using L4 GPUs

In this guide, we will introduce how to use vLLM serve Llama base model, pretrained model and Lora adapter model on GKE.


## Prerequisites
*   A terminal with `kubectl` and `gcloud` installed. Cloud Shell works great!
*   L4 GPUs quota to be able to run additional 2 L4 GPUs
*   You have downloaded a llama2 base model,  pretrained model or Lora adapter model on GCS bucket

## Build vllm docker image
Let’s start by setting a few environment variables that will be used throughout this post. You should modify these variables to meet your environment and needs. 

Run the following commands to set the env variables and make sure to replace `<my-project-id>` and replace `<vpc_subnet>` if you are using a manually vpc:
```bash
gcloud config set project <my-project-id>
export PROJECT_ID=$(gcloud config get project)
export REGION=us-central1
export SUBNET=<vpc_subnet>
```

Then you can build and push the image using cloud build
```bash
gcloud auth configure-docker ${REGION}-docker.pkg.dev

# create vllm artifacts id you do not have
gcloud artifacts repositories create artifact-vllm \
    --repository-format=docker \
    --location=${REGION} \
    --description="vllm repo" \
    --immutable-tags \
    --async

gcloud builds submit --tag ${REGION}-docker.pkg.dev/${PROJECT_ID}/artifact-vllm/vllm-server:0.1
```

## Creating the GKE cluster with L4 nodepools

Set the bucket name to your bucket that have base model/fine tune model.
```bash
export BUCKET_NAME=<my-gcs-model-bucket>
export SERVICE_ACCOUNT="vllm-l4@${PROJECT_ID}.iam.gserviceaccount.com"
```

Create the GKE cluster by running:
```bash
gcloud container clusters create vllm-l4 --location ${REGION} \
  --accelerator type=nvidia-l4,count=2,gpu-driver-version=latest \
  --machine-type g2-standard-24 \
  --ephemeral-storage-local-ssd=count=2 \
  --workload-pool ${PROJECT_ID}.svc.id.goog \
  --enable-image-streaming --enable-shielded-nodes \
  --shielded-secure-boot \
  --shielded-integrity-monitoring \
  --num-nodes=1 \
  --enable-autoscaling \
  --min-nodes=0 \
  --max-nodes=2 \
  --enable-ip-alias \
  --node-locations=${REGION}-a \
  --subnetwork=${SUBNET} \
  --workload-pool=${PROJECT_ID}.svc.id.goog \
  --labels="ai-on-gke=vllm-l4" \
  --addons GcsFuseCsiDriver
```

The default nodepool of g2-standard-24 has been created. The default node number is 1, and is scaled down to 0 nodes, up to 2 nodes. So you are not paying for any GPUs until you start launching Kubernetes Pods that request GPUs.

## Deploy vllm serve

### Configuring GCS and required permissions

Let’s create a Google Service Account that has read permissions to the GCS bucket. Then create a Kubernetes Service Account named `vllm-l4` that is able to use the Google Service Account.

To do this, first create a new Google Service Account:
```bash
gcloud iam service-accounts create vllm-l4
```

Assign the required GCS permissions to the Google Service Account:
```bash
gcloud storage buckets add-iam-policy-binding gs://${BUCKET_NAME} \
  --member="serviceAccount:${SERVICE_ACCOUNT}" --role=roles/storage.admin
```

Allow the Kubernetes Service Account `vllm-l4` in the `default` namespace to use the Google Service Account:
```bash
gcloud iam service-accounts add-iam-policy-binding ${SERVICE_ACCOUNT} \
  --role roles/iam.workloadIdentityUser \
  --member "serviceAccount:${PROJECT_ID}.svc.id.goog[default/vllm-l4]"
```

Create a new Kubernetes Service Account:
```bash
kubectl create serviceaccount vllm-l4
kubectl annotate serviceaccount vllm-l4 iam.gke.io/gcp-service-account=vllm-l4@${PROJECT_ID}.iam.gserviceaccount.com
```


### Create deployment

Create a file named vllm.yaml with the following content. 

Inside the YAML file the following settings are used:

- **TENSOR_PARALLEL_SIZE**, optional. Setting to the number of GPUs to distributly inference and serve.   This has to be set to 2 because 2 x NVIDIA L4 GPUs are used.
- **MODEL_GCS_URI**, optional. Setting to the gcs uri of llama2 base model, pretain model or full-paremeter fine tuned model. If you do not specify it, vllm will use default model: facebook/opt-125m
- **PEFT_MODEL_GCS_URI**, optional. Setting to the gcs uri of lora adapter model. vllm do not support lora adapter, so we need to merge base model and lora adapter firstly.

[embedmd]:# (vllm_serve.yaml)
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: vllm-l4
spec:
  replicas: 1
  selector:
    matchLabels:
      app: vllm-l4
  template:
    metadata:
      labels:
        app: vllm-l4
      annotations:
        kubectl.kubernetes.io/default-container: vllm-l4
        gke-gcsfuse/volumes: "true"
        gke-gcsfuse/memory-limit: 400Mi
        gke-gcsfuse/ephemeral-storage-limit: 30Gi
    spec:
      terminationGracePeriodSeconds: 60
      containers:
      - name: vllm-l4
        image: ${REGION}-docker.pkg.dev/${PROJECT_ID}/artifact-vllm/vllm-server:0.1
        command: ["python3", "/root/scripts/launcher.py", "--tensor_parallel_size=$(TENSOR_PARALLEL_SIZE)", "--model_gcs_uri=$(MODEL_GCS_URI)", "--peft_model_gcs_uri=$(PEFT_MODEL_GCS_URI)"]
        resources:
          limits:
            nvidia.com/gpu: 2
        env:
        - name: TENSOR_PARALLEL_SIZE
          value: "2"
        - name: MODEL_GCS_URI
          value: gs://${BUCKET_NAME}/llama-2-7b-chat-hf
        - name: PEFT_MODEL_GCS_URI
          value: gs://${BUCKET_NAME}/peft_model
        volumeMounts:
        - mountPath: /dev/shm
          name: dshm
        - name: gcs-fuse-csi-ephemeral
          mountPath: /gcs-mount
      serviceAccountName: vllm-l4
      volumes:
      - name: dshm
        emptyDir:
          medium: Memory
          sizeLimit: 48G
      - name: gcs-fuse-csi-ephemeral
        csi:
          driver: gcsfuse.csi.storage.gke.io
          volumeAttributes:
            bucketName: ${BUCKET_NAME}
            mountOptions: "implicit-dirs"
      nodeSelector:
        cloud.google.com/gke-accelerator: nvidia-l4
```

Create the deployment for serving:
```bash
kubectl apply -f vllm_serve.yaml
```

Check the logs and make sure there are no errors:
```bash
kubectl logs -l app=vllm-l4
```

The deployment will take about 30 mins to complete, have a coffee and wait the logs show below information:
```bash
2023-11-26 14:47:34,978 INFO worker.py:1673 -- Started a local Ray instance.
INFO 11-26 14:47:36 llm_engine.py:72] Initializing an LLM engine with config: model='/gcs-mount/peft_merged_model', tokenizer='/gcs-mount/peft_merged_model', tokenizer_mode=auto, revision=None, trust_remote_code=False, dtype=torch.float16, max_seq_len=2048, download_dir=None, load_format=auto, tensor_parallel_size=2, quantization=None, seed=0)
Special tokens have been added in the vocabulary, make sure the associated word embeddings are fine-tuned or trained.
INFO 11-26 14:47:55 llm_engine.py:205] # GPU blocks: 3327, # CPU blocks: 1024
INFO:     Started server process [49]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
```


### Create Service

After the vllm serve deployment was done, you can use the vllm_service.yaml to expose it to a service.

[embedmd]:# (vllm_lb_service.yaml)
```yaml
apiVersion: v1
kind: Service
metadata:
  name: vllm-lb-service
spec:
  type: LoadBalancer
  selector:
    app: vllm-l4
  ports:
  - protocol: TCP
    port: 8000
    targetPort: 8000
```

Create the deployment for serving:
```bash
kubectl apply -f vllm_lb_service.yaml
```

Check the logs and make sure there are no errors:
```bash
kubectl get service vllm_lb_service
```

you will see the output contains lb external ip.
```bash
NAME            TYPE           CLUSTER-IP     EXTERNAL-IP      PORT(S)          AGE
vllm-lb-service   LoadBalancer   <CLUSTER-IP>   <EXTERNAL-IP>   8000:30875/TCP   2m31s
```



### Inference request
Now you can chat with your model through a simple curl:
```bash
curl http://load_balancer_ip:8000/generate \
    -d '{
        "prompt": "San Francisco is a",
        "use_beam_search": true,
        "n": 4,
        "temperature": 0
    }'
```

### vLLM benchmark inside pod

Login the pod
```bash
kubectl exec -iy pod-name bash
```

Run benchmark test using vllm test script

```bash
python3 /root/vllm/benchmarks/benchmark_serving.py --backend vllm --tokenizer /gcs-mount/peft_merged_model --dataset /root/datasets/ShareGPT_V3_unfiltered_cleaned_split.json --host 0.0.0.0 --port 8000
```
