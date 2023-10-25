#!/bin/bash

# Entrypoint of SD training docker container.
# Discover multi nodes topology and call trainer.

# example:
# CLUSTER_SPEC='{"cluster":{"workerpool0":["cmle-training-workerpool0-4c1d6d97d0-0:2222"],"workerpool1":["cmle-training-workerpool1-4c1d6d97d0-0:2222","cmle-training-workerpool1-4c1d6d97d0-1:2222","cmle-training-workerpool1-4c1d6d97d0-2:2222"]},"environment":"cloud","task":{"type":"workerpool1","index":3},"job":"{\"python_module\":\"\",\"package_uris\":[],\"job_args\":[]}","open_ports":[3333]}'

# NCCL setup
# export NCCL_DEBUG=INFO
# export NCCL_NSOCKS_PERTHREAD=1
# export NCCL_SOCKET_NTHREADS=2
# export NCCL_MIN_NCHANNELS=1

# huggingface setup
# export HF_HUB_OFFLINE=1
#export HF_HUB_DISABLE_TELEMETRY=1

#
# Maps the Vertex CustomJob GCS locations to the local FUSE dir supported by Vertex.
# 
if [[ -z ${OUTPUT_FOLDER} ]]; then OUTPUT_FOLDER=$(echo ${AIP_MODEL_DIR} | sed -E -e 's|^gs:/|/gcs|' -e 's|/$||'); fi
if [[ -z ${CHKPTS_FOLDER} ]]; then CHKPTS_FOLDER=$(echo ${AIP_CHECKPOINT_DIR} | sed -E -e 's|^gs:/|/gcs|' -e 's|/$||'); fi
if [[ -z ${JOBLOG_FOLDER} ]]; then JOBLOG_FOLDER=$(echo ${AIP_TENSORBOARD_LOG_DIR} | sed -E -e 's|^gs:/|/gcs|' -e 's|/$||'); fi
if [ ! -d "/gcs" ]; then # Create the folders under test conditions
    OUTPUT_FOLDER="."${OUTPUT_FOLDER}; mkdir -p ${OUTPUT_FOLDER}
    CHKPTS_FOLDER="."${CHKPTS_FOLDER}; mkdir -p ${CHKPTS_FOLDER}
    JOBLOG_FOLDER="."${JOBLOG_FOLDER}; mkdir -p ${JOBLOG_FOLDER}
fi

if [ -n "${JOBLOG_FOLDER}" ]; then

    OTHER_OPTIONS=${OTHER_OPTIONS}" \
    --enable_tensorboard \
    --tensorboard_path ${JOBLOG_FOLDER}"

fi

if [[ -n "$SERVING_CONTAINER_URI" ]]; then

    OTHER_OPTIONS=${OTHER_OPTIONS}" \
    --serving_container_image_uri $SERVING_CONTAINER_URI \
    --project $CLOUD_ML_PROJECT_ID \
    --location $CLOUD_ML_REGION"

fi

if [[ -z $CLUSTER_SPEC ]]

then
    echo "========== Launch on local machine =========="
    
    set -x
    torchrun --nnodes=1 --node_rank=0 \
    --nproc_per_node=$NUM_GPU_PER_NODE \
    main.py \
    --data_path $DATA_PATHS \
    --data_split 10,0,0 \
    --model_name_or_path $MODEL_PATH \
    --per_device_train_batch_size $PER_DEVICE_BATCH_SIZE \
    --per_device_eval_batch_size $PER_DEVICE_BATCH_SIZE \
    --max_seq_len 512 \
    --learning_rate 9.65e-6 \
    --weight_decay 0. \
    --num_train_epochs 1  \
    --gradient_accumulation_steps 1 \
    --lr_scheduler_type cosine \
    --num_warmup_steps 0 \
    --seed 1234 \
    --gradient_checkpointing \
    --zero_stage $ZERO_STAGE \
    --deepspeed \
    --lora_dim 128 \
    --lora_module_name "layers." \
    --only_optimize_lora \
    --output_dir ${OUTPUT_FOLDER} \
    --print_loss \
    ${OTHER_OPTIONS}

else
    echo "========== Launch on cloud =========="
    echo "CLUSTER_SPEC:" $CLUSTER_SPEC
    
    primary_node=`echo $CLUSTER_SPEC | jq -r '.cluster.workerpool0[0]'`
    
    IFS=':' read -ra primary_node_split <<< $primary_node
    primary_node_addr=${primary_node_split[0]}
    primary_node_port=${primary_node_split[1]}

    workerpool=`echo $CLUSTER_SPEC | jq -r '.task.type'`
    if [[ $workerpool = "workerpool0" ]]
    then
        node_rank=0
    else
        node_rank=`echo $CLUSTER_SPEC | jq -r '.task.index'`
        node_rank=$(($node_rank + 1))
    fi
    workerpool1_nodes=`echo $CLUSTER_SPEC | jq -r '.cluster.workerpool1 | length'`
    num_nodes=$(($workerpool1_nodes + 1))
    
    echo "primary node address: " $primary_node_addr
    echo "primary node port: " $primary_node_port
    echo "num nodes: " $num_nodes
    echo "node rank: " $node_rank
    
    if [[ $num_nodes = 1 ]]
    then
        set -x
        torchrun --nnodes=1 --node_rank=0 \
        --nproc_per_node=$NUM_GPU_PER_NODE \
        main.py \
        --data_path $DATA_PATHS \
        --data_split 10,0,0 \
        --model_name_or_path $MODEL_PATH \
        --per_device_train_batch_size $PER_DEVICE_BATCH_SIZE \
        --per_device_eval_batch_size $PER_DEVICE_BATCH_SIZE \
        --max_seq_len 512 \
        --learning_rate 9.65e-6 \
        --weight_decay 0. \
        --num_train_epochs 1  \
        --gradient_accumulation_steps 1 \
        --lr_scheduler_type cosine \
        --num_warmup_steps 0 \
        --seed 1234 \
        --gradient_checkpointing \
        --zero_stage $ZERO_STAGE \
        --deepspeed \
        --lora_dim 128 \
        --lora_module_name "layers." \
        --only_optimize_lora \
        --output_dir ${OUTPUT_FOLDER} \
        --print_loss \
        ${OTHER_OPTIONS}

    else
        set -x
        torchrun \
        --nnodes=$num_nodes \
        --node_rank=$node_rank \
        --master_addr=$primary_node_addr \
        --master_port=$primary_node_port \
        --nproc_per_node=$NUM_GPU_PER_NODE \
        main.py \
        --data_path $DATA_PATHS \
        --data_split 10,0,0 \
        --model_name_or_path $MODEL_PATH \
        --per_device_train_batch_size $PER_DEVICE_BATCH_SIZE \
        --per_device_eval_batch_size $PER_DEVICE_BATCH_SIZE \
        --max_seq_len 512 \
        --learning_rate 9.65e-6 \
        --weight_decay 0. \
        --num_train_epochs 1  \
        --gradient_accumulation_steps 1 \
        --lr_scheduler_type cosine \
        --num_warmup_steps 0 \
        --seed 1234 \
        --gradient_checkpointing \
        --zero_stage $ZERO_STAGE \
        --deepspeed \
        --lora_dim 128 \
        --lora_module_name "layers." \
        --only_optimize_lora \
        --output_dir ${OUTPUT_FOLDER} \
        --print_loss \
        ${OTHER_OPTIONS}
    fi

fi