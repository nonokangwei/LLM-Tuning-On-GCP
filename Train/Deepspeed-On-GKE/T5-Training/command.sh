deepspeed --hostfile=./hostfile run_seq2seq_deepspeed-args.py --batch_size=8 --epoch=1 --train_dataset_path=/gcs/lsj-public/deepspeed/data/train --test_dataset_path=/gcs/lsj-public/deepspeed/data/eval --aip_model_dir=GCS_MODEL_DIR --aip_tensorboard_log_dir=GCS_LOG_DIR

env >> /etc/environment;exit
deepspeed --hostfile=/config/hostfile run.py --batch_size=8 --epoch=1 --train_dataset_path=/gcs/deepspeed/data/train --test_dataset_path=/gcs/deepspeed/data/eval --aip_model_dir=gs://flius-vpc-2-bucket/deepspeed/model --aip_tensorboard_log_dir=gs://flius-vpc-2-bucket/deepspeed/log
tensorboard dev upload --logdir ./log
https://tensorboard.dev/experiment/7BxOUMTZQe63JwOMzfGRyQ/#scalars&runSelectionState=eyIuIjp0cnVlfQ%3D%3D
https://tensorboard.dev/experiment/LtbzWF1cQkSNNHDxrDVxwA/#scalars
#2vm gke
https://tensorboard.dev/experiment/5r6tgkj0QYmK3PIYyjLKuQ/#scalars

events.out.tfevents.1693321420.deepspeed-shijun-0.132.0


deepspeed --hostfile=/config/hostfile run.py --batch_size=8 --epoch=1 --train_dataset_path=/gcs/deepspeed/data/train --test_dataset_path=/gcs/deepspeed/data/eval --aip_model_dir=gs://flius-vpc-2-bucket/deepspeed/model --aip_tensorboard_log_dir=./log
nohup deepspeed --hostfile=/config/hostfile run.py --batch_size=8 --epoch=1 --train_dataset_path=/gcs/deepspeed/data/train --test_dataset_path=/gcs/deepspeed/data/eval --model_output_dir=/model --tensorboard_log_dir=/log &
deepspeed --hostfile=./hostfile run.py --batch_size=8 --epoch=1 --train_dataset_path=./gcs/data/train --test_dataset_path=./gcs/data/eval --model_output_dir=/model --tensorboard_log_dir=/log

deepspeed --hostfile=/config/hostfile run.py --batch_size=8 --epoch=1 --train_dataset_path=/gcs/deepspeed/data/train --test_dataset_path=/gcs/deepspeed/data/eval --model_output_dir=/model --tensorboard_log_dir=/log

gcloud container clusters update tcpx-cluster \
    --update-addons GcsFuseCsiDriver=ENABLED \
    --region=us-central1

kubectl create serviceaccount fuse-sa \
    --namespace default

kubectl annotate serviceaccount fuse-sa \
    --namespace default \
    iam.gke.io/gcp-service-account=fuse-gsa@flius-vpc-2.iam.gserviceaccount.com

gcloud container node-pools update pool-1 --cluster=tcpx-cluster --enable-gvnic --enable-fast-socket --region=us-central1

kubectl apply -f https://raw.githubusercontent.com/GoogleCloudPlatform/container-engine-accelerators/master/nvidia-driver-installer/cos/daemonset-preloaded-latest.yaml


gcloud beta container --project "flius-vpc-2" clusters create "tcp-cluster-3" --region "us-central1" --no-enable-basic-auth --cluster-version "1.26.5-gke.1400" --release-channel "None" --machine-type "e2-standard-4" --image-type "COS_CONTAINERD" --disk-type "pd-balanced" --disk-size "100" --metadata disable-legacy-endpoints=true --scopes "https://www.googleapis.com/auth/cloud-platform" --num-nodes "1" --enable-ip-alias --network "projects/flius-vpc-2/global/networks/tcpx-net-1" --subnetwork "projects/flius-vpc-2/regions/us-central1/subnetworks/tcpx-sub-1" --no-enable-intra-node-visibility --default-max-pods-per-node "110" --security-posture=disabled --workload-vulnerability-scanning=disabled --no-enable-master-authorized-networks --addons HorizontalPodAutoscaling,HttpLoadBalancing,GcePersistentDiskCsiDriver,GcsFuseCsiDriver --enable-autoupgrade --enable-autorepair --max-surge-upgrade 1 --max-unavailable-upgrade 0 --no-enable-managed-prometheus --workload-pool "flius-vpc-2.svc.id.goog" --enable-shielded-nodes --enable-image-streaming --node-locations "us-central1-a"

gcloud container node-pools create pool-3 \
    --region=us-central1 \
    --node-version=1.26.2-gke.1000 \
    --num-nodes=2 \
    --workload-metadata=GKE_METADATA \
    --accelerator type=nvidia-tesla-t4,count=2 \
    --machine-type=n1-standard-32 \
    --cluster=tcpx-cluster \
    --enable-fast-socket \
    --enable-gvnic \
    --disk-size=200
# manually install fast socket
https://github.com/GoogleCloudPlatform/container-engine-accelerators/blob/master/fast-socket-installer/fast-socket-installer.yaml

gcloud container node-pools create pool-4 \
    --region=us-central1 \
    --node-version=1.26.2-gke.1000 \
    --num-nodes=2 \
    --workload-metadata=GKE_METADATA \
    --accelerator type=nvidia-tesla-t4,count=2 \
    --machine-type=n1-standard-32 \
    --cluster=tcpx-cluster \
    --enable-gvnic \
    --disk-size=200

    gcloud container node-pools create pool-3 \
        --region=us-central1 \
        --num-nodes=1 \
        --workload-metadata=GKE_METADATA \
        --accelerator type=nvidia-tesla-t4,count=2 \
        --machine-type=n1-standard-32 \
        --cluster=tcpx-cluster \
        --enable-fast-socket \
        --enable-gvnic \
        --disk-size=200

        gcloud container node-pools create pool-8 \
            --region=us-central1 \
            --num-nodes=1 \
            --workload-metadata=GKE_METADATA \
            --accelerator type=nvidia-tesla-t4,count=2 \
            --machine-type=n1-standard-32 \
            --cluster=default-cluster-dpv2 \
            --enable-fast-socket \
            --enable-gvnic \
            --disk-size=200

sudo docker run --net=host --shm-size=16g --hostname=deepspeed-train --volume ./gcs:/gcs --gpus all -it  us-docker.pkg.dev/flius-vpc-2/flius-vpc-2-repo/deepspeed:fromshijun-base bash

192.168.1.51 slots=2
192.168.1.53 slots=2
192.168.1.54 slots=2
192.168.1.55 slots=2
