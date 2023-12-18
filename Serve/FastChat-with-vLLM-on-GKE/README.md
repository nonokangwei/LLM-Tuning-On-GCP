# Serve Llama 7b using FastChat as front api serer, vLLM as backend on GKE L4 GPUs

In this guide, we will introduce how to serve Llama base model, pretrained model and Lora adapter model on GKE. We will use FastChat as the front Chatbot api server, vLLM as backend serving server.


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

gcloud builds submit --tag ${REGION}-docker.pkg.dev/${PROJECT_ID}/artifact-vllm/vllm-server:fastchat
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

### (option) Download LLama2 7b model
If you want to deploy a model directly without going through the Train part of this solution, you can download the model in this step.
If you have compeleted Train part, you can go through without this step.

Create a Secret to store your HuggingFace token which will be used by the Kubernetes job:
```bash
kubectl create secret generic vllm-l4 \
  --from-literal="HF_TOKEN=<paste-your-own-token>"
```

Let's use Kubernetes Job to download the Llama 2 7B model from HuggingFace. The file download-model.yaml in this repo shows how to do this:
[embedmd]:# (download_llama2-7b.yaml)
```yaml
apiVersion: batch/v1
kind: Job
metadata:
  name: model-loader
  namespace: default
spec:
  template:
    metadata:
      annotations:
        kubectl.kubernetes.io/default-container: loader
        gke-gcsfuse/volumes: "true"
        gke-gcsfuse/memory-limit: 400Mi
        gke-gcsfuse/ephemeral-storage-limit: 30Gi
    spec:
      restartPolicy: OnFailure
      containers:
      - name: loader
        image: python:3.11
        command:
        - /bin/bash
        - -c
        - |
          pip install huggingface_hub
          mkdir -p /gcs-mount/llama2-7b
          python3 - << EOF
          from huggingface_hub import snapshot_download
          model_id="meta-llama/Llama-2-7b-hf"
          snapshot_download(repo_id=model_id, local_dir="/gcs-mount/llama2-7b",
                            local_dir_use_symlinks=False, revision="main",
                            ignore_patterns=["*.safetensors", "model.safetensors.index.json"])
          EOF
        imagePullPolicy: IfNotPresent
        env:
        - name: HUGGING_FACE_HUB_TOKEN
          valueFrom:
            secretKeyRef:
              name: vllm-l4
              key: HF_TOKEN
        volumeMounts:
        - name: gcs-fuse-csi-ephemeral
          mountPath: /gcs-mount
      serviceAccountName: vllm-l4
      volumes:
      - name: gcs-fuse-csi-ephemeral
        csi:
          driver: gcsfuse.csi.storage.gke.io
          volumeAttributes:
            bucketName: ${BUCKET_NAME}
            mountOptions: "implicit-dirs"
```

Create the deployment to download the model:
```bash
kubectl apply -f download-llama2-7b.yaml
```


### Create controller deployment

To serve using the fastchat web UI, you need three main components: web servers that interface with users, model workers that host one or more models, and a controller to coordinate the webserver and model workers. 

Launch the controller

[embedmd]:# (fastchat-controller.yaml)
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  creationTimestamp: null
  labels:
    app: controller
  name: controller
spec:
  replicas: 1
  selector:
    matchLabels:
      app: controller
  template:
    metadata:
      labels:
        app: controller
    spec:
      containers:
      - image: ${REGION}-docker.pkg.dev/${PROJECT_ID}/artifact-vllm/vllm-server:fastchat
        name: controller
        command:
          - python3
          - -m
          - fastchat.serve.controller
          - --host
          - "0.0.0.0"
          - --port
          - "21001"
        ports:
        - containerPort: 21001
      nodeSelector:
        cloud.google.com/gke-nodepool: default-pool

---

apiVersion: v1
kind: Service
metadata:
  name: controller-svc
spec:
  ports:
  - port: 21001
    protocol: TCP
    targetPort: 21001
  selector:
    app: controller
  type: ClusterIP
```

Create the deployment for serving:
```bash
kubectl apply -f fastchat-controller.yaml
```

### Create model worker deployment

To serve using the fastchat web UI, you need three main components: web servers that interface with users, model workers that host one or more models, and a controller to coordinate the webserver and model workers. 

Launch the controller

[embedmd]:# (fastchat-controller.yaml)
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  creationTimestamp: null
  labels:
    app: controller
  name: controller
spec:
  replicas: 1
  selector:
    matchLabels:
      app: controller
  template:
    metadata:
      labels:
        app: controller
    spec:
      containers:
      - image: ${REGION}-docker.pkg.dev/${PROJECT_ID}/artifact-vllm/vllm-server:fastchat
        name: controller
        command:
          - python3
          - -m
          - fastchat.serve.controller
          - --host
          - "0.0.0.0"
          - --port
          - "21001"
        ports:
        - containerPort: 21001
      nodeSelector:
        cloud.google.com/gke-nodepool: default-pool

---

apiVersion: v1
kind: Service
metadata:
  name: controller-svc
spec:
  ports:
  - port: 21001
    protocol: TCP
    targetPort: 21001
  selector:
    app: controller
  type: ClusterIP
```

Create the deployment for serving:
```bash
kubectl apply -f fastchat-controller.yaml
```

### Create FastChat Gradio web server deployment

Launch the Gradio web server

[embedmd]:# (fastchat-controller.yaml)
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  creationTimestamp: null
  labels:
    app: gui
  name: gui
spec:
  replicas: 1
  selector:
    matchLabels:
      app: gui
  template:
    metadata:
      labels:
        app: gui
    spec:
      containers:
      - image: ${REGION}-docker.pkg.dev/${PROJECT_ID}/artifact-vllm/vllm-server:fastchat
        name: gui
        command:
          - python3
          - -m
          - fastchat.serve.gradio_web_server
          - --controller
          - http://controller-svc:21001        
        ports:
        - containerPort: 7860
      nodeSelector:
        cloud.google.com/gke-nodepool: default-pool        

---
apiVersion: v1
kind: Service
metadata:
  name: gui-svc
spec:
  ports:
  - port: 80
    protocol: TCP
    targetPort: 7860
  selector:
    app: gui
  type: LoadBalancer

```

Create controller
```
kubectl apply -f fastchat-model-worker.yaml
```

## Test GUI
1. Get external IP address
```
kubectl get svc gui-svc
```
2. Open browser and input external ip address

