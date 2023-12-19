#!/bin/bash

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

# get current time
now=$(date +%s)

# calculate time after 24 hours
after=$((now + 86400))

# sleep 24 hours
until [ $(date +%s) -ge $after ]; do
    sleep 1
done

echo "24 hours already!"