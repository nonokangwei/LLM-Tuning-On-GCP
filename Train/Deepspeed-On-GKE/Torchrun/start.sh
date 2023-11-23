cp /etc/ssh/ssh_config .
chmod 666 ssh_config
echo "StrictHostKeyChecking no" > ssh_config
sudo cp ssh_config /etc/ssh/ssh_config
sudo service ssh start
env >> /etc/environment

if [[ ${JOB_COMPLETION_INDEX} == 0 ]];
  then echo ${MY_POD_IP} > /gcs/master.txt
fi

while true; do

  if [ -f "/gcs/master.txt" ]; then
    export master_addr=$(cat /gcs/master.txt)

    torchrun --nnodes=4 --node_rank=${JOB_COMPLETION_INDEX} --master_addr=${master_addr} --master_port=1111 --nproc_per_node=2 main.py --data_path "samsum" --data_split 10,0,0 --model_name_or_path "/gcs/deepspeed_repo/base_model/Llama-2-7b-hf/Llama-2-7b-hf" --per_device_train_batch_size 4 --per_device_eval_batch_size 4 --max_seq_len 512 --learning_rate 9.65e-6 --weight_decay 0. --num_train_epochs 1  --gradient_accumulation_steps 1 --lr_scheduler_type cosine --num_warmup_steps 0 --seed 1234 --gradient_checkpointing --zero_stage 3 --deepspeed --output_dir output --print_loss
    break
  else
    sleep 10
  fi
done

if [ -f "/gcs/master.txt" ]; then
  rm -rf /gcs/master.txt
fi
