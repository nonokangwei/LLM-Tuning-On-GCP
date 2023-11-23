# User Guide
## Clone this repo to your local
```
git clone https://github.com/Leisureroad/deepspeed-on-gke
cd deepspeed-on-gke/DeepSpeed-Chat
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

## Use cloud build to build and push docker image to artifacts registry.
```
PROJECT_ID=<your project id>
AR_REPO=<artifacts registry repo name>
gcloud builds submit --tag us-docker.pkg.dev/${PROJECT_ID}/${AR_REPO}/deepspeed-chat:fromshijun-base
```

## Apply kubernetes manifests
```
kubectl apply -f statefulset-deepspeed-chat.yaml
```

## Run deepspeed inside a pod
```
deepspeed --hostfile=/config/hostfile main.py \
   --sft_only_data_path "samsum" \
   --model_name_or_path "meta-llama/Llama-2-7b-hf" \
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
