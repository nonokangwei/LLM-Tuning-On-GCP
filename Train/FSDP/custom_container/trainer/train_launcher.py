import argparse
import sys
import os
import json
import socket
import torch.distributed.run as training_distributed_launch
import fire
from llama_recipes.finetuning import main as llama_main
from google.cloud import logging as google_logging
from google.cloud.logging_v2.handlers import setup_logging
from google.cloud.logging.handlers import CloudLoggingHandler
from google.cloud import logging_v2
import logging
from torch.distributed.elastic.agent.server.api import (
    SimpleElasticAgent, 
    _get_socket_with_port, 
    _get_fq_hostname
)
from contextlib import closing

# Get master ip and port from env CLUSTER_SPEC
def GetMasterInfor(worker_pool_spec: str = None):
    if worker_pool_spec == None:
        cluster_spec_str = os.getenv("CLUSTER_SPEC")
        cluster_spec_dict = json.loads(cluster_spec_str)
    else:
        cluster_spec_dict = json.loads(worker_pool_spec)
    
    master_ip = cluster_spec_dict["cluster"]["workerpool0"][0].split(":")[0]
    master_port = cluster_spec_dict["open_ports"][0]
    return master_ip, master_port

# Get job id from env CLOUD_ML_JOB_ID    
def GetJobID():
    job_id = os.getenv("CLOUD_ML_JOB_ID")
    return job_id

# Get project number from env CLOUD_ML_PROJECT_ID
def GetProjectNumber():
    project_id = os.getenv("CLOUD_ML_PROJECT_ID")
    return project_id

# Get node id from env CLUSTER_SPEC
def GetNodeInfor(worker_pool_spec: str = None):
    if worker_pool_spec == None:
        cluster_spec_str = os.getenv("CLUSTER_SPEC")
        cluster_spec_dict = json.loads(cluster_spec_str)
    else:
        cluster_spec_dict = json.loads(worker_pool_spec)
    
    node_pool = cluster_spec_dict["task"]["type"]
    node_index = cluster_spec_dict["task"]["index"]
    node_id = cluster_spec_dict["cluster"][node_pool][int(node_index)]
    return node_id

# Get node role from env CLUSTER_SPEC
def GetNodeRole(worker_pool_spec: str = None):
    if worker_pool_spec == None:
        cluster_spec_str = os.getenv("CLUSTER_SPEC")
        cluster_spec_dict = json.loads(cluster_spec_str)
    else:
        cluster_spec_dict = json.loads(worker_pool_spec)
    
    node_pool = cluster_spec_dict["task"]["type"]
    node_index = cluster_spec_dict["task"]["index"]
    node_role = node_pool + "-" + str(node_index)
    return node_role

def _hook_set_master_addr_port(store, master_addr, master_port, local_addr):
    if master_port is None:
        sock = _get_socket_with_port()
        with closing(sock):
            master_port = sock.getsockname()[1]

    if master_addr is None:
        if local_addr:
            master_addr = local_addr
        else:
            hostname = _get_fq_hostname()
            # use IP address as master_addr
            master_addr = socket.gethostbyname(hostname)

    store.set("MASTER_ADDR", master_addr.encode(encoding="UTF-8"))
    store.set("MASTER_PORT", str(master_port).encode(encoding="UTF-8"))

if __name__ == '__main__':
    # Init logging
    logging_client = google_logging.Client(project=GetProjectNumber())
    log_name = "ml.googleapis.com%2F" + GetJobID()
    resource = logging_v2.Resource(type="ml_job", labels={"job_id": GetJobID(), "task_name": "service", "project_id": GetProjectNumber(), "node_id": GetNodeInfor()})
    handler = CloudLoggingHandler(client=logging_client, resource=resource, name=log_name, labels={"node_id": GetNodeInfor()})
    setup_logging(handler)
    sys.stderr.write = logging.error
    sys.stdout.write = logging.info 

    # Init args parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--nnodes', dest='nnodes', type=int, default=None)
    parser.add_argument('--nproc_per_node', dest='nproc_per_node', type=int, default=None)
    # Parse args
    cmd_args, _ = parser.parse_known_args()
    # Parse orginal args
    args = sys.argv[1:]

    if cmd_args.nnodes != None and cmd_args.nproc_per_node != None:
        # Check if multi node training
        if cmd_args.nnodes > 1:
            head_node_ip, head_node_port = GetMasterInfor()
            if GetNodeRole() == "workerpool0-0":
                rdzv_endpoint = socket.gethostbyname(socket.gethostname()) + ":" + str(head_node_port)
            else:
                rdzv_endpoint = head_node_ip + ":" + str(head_node_port)
            # Add param rdzv_id
            args.insert(0, GetJobID())
            args.insert(0, "--rdzv-id")
            # Add param rdzv_backend
            args.insert(0, "c10d")
            args.insert(0, "--rdzv-backend")
            # Add param rdzv_endpoint
            args.insert(0, rdzv_endpoint)
            args.insert(0, "--rdzv-endpoint")
            
            setattr(SimpleElasticAgent, "_set_master_addr_port", staticmethod(_hook_set_master_addr_port))
        # Launch distributed training
        # Sample: python3 train_launcher.py --nnodes 2 --nproc_per_node 1 ./finetune.py --enable_fsdp --use_peft --peft_method lora --model_name /gcs/llama2ft-project-kangwe-poc-unique/llama-7b --pure_bf16 --output_dir /root/save/model
        training_distributed_launch.main(args)
    else:
        # Launch single node training
        # Sample: python3 train_launch.py ./finetune.py --enable_fsdp --use_peft --peft_method lora --model_name /gcs/llama2ft-project-kangwe-poc-unique/llama-7b --pure_bf16 --output_dir /root/save/model
        args.append("--nnodes")
        args.append("1")
        args.append("--nproc_per_node")
        args.append("1")
        training_distributed_launch.main(args)

# ./finetune.py --nnodes 2 --nproc_per_node 1 --rdzv_id demo --rdzv_backend c10d --rdzv_endpoint cmle-training-workerpool0-0dde96e0c0-0:2222 --enable_fsdp --use_peft --peft_method lora --model_name /gcs/llama2ft-project-kangwe-poc-unique/llama-7b --pure_bf16 --output_dir /root/save/model
