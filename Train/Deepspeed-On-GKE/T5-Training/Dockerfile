#FROM europe-docker.pkg.dev/vertex-ai/training/pytorch-gpu.1-10:latest
FROM pytorch/pytorch:2.0.0-cuda11.7-cudnn8-devel

WORKDIR /

LABEL com.nvidia.volumes.needed=nvidia_driver

# env variables for proper GPU setup
ENV PATH=/opt/conda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility
ENV LD_LIBRARY_PATH=/usr/local/nvidia/lib:/usr/local/nvidia/lib64
#RUN export PATH="/usr/local/nvidia/bin:/usr/local/cuda/bin:${PATH}"
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && \
        apt-get install -y --no-install-recommends \
        software-properties-common build-essential autotools-dev \
        nfs-common pdsh \
        cmake g++ gcc \
        apt-utils jq gnupg \
        curl wget vim tmux emacs less unzip git\
        htop iftop iotop ca-certificates openssh-client openssh-server \
        rsync iputils-ping net-tools sudo \
        llvm-9-dev

RUN apt-get install -y --no-install-recommends libaio1
RUN apt-get install -y --no-install-recommends libaio-dev

# Install reduction server plugin.
#RUN echo "deb https://packages.cloud.google.com/apt google-fast-socket main" | tee /etc/apt/sources.list.d/google-fast-socket.list && \
#    curl -s -L https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key add - && \
#    apt update && apt install -y google-reduction-server

#ssh
ENV SSH_PORT=2222
RUN cat /etc/ssh/sshd_config > ${STAGE_DIR}/sshd_config && \
        sed "0,/^#Port 22/s//Port ${SSH_PORT}/" ${STAGE_DIR}/sshd_config > /etc/ssh/sshd_config
RUN sed -i 's/PermitRootLogin without-password/PermitRootLogin yes/' /etc/ssh/sshd_config
RUN sed 's@session\\s*required\\s*pam_loginuid.so@session optional pam_loginuid.so@g' -i /etc/pam.d/sshd

# RUN sudo service ssh start
#RUN ssh-keygen -t rsa
ADD id_rsa /root/.ssh/id_rsa
RUN chmod 0400 /root/.ssh/id_rsa
RUN touch /root/.ssh/config && \
    printf "Port 2222\n StrictHostKeyChecking no\n  UserKnownHostsFile=/dev/null" > /root/.ssh/config
RUN chmod 600 /root/.ssh/config
ADD id_rsa.pub /root/.ssh/id_rsa.pub
ADD authorized_keys /root/.ssh/authorized_keys
RUN chmod 600 /root/.ssh/config
RUN chmod 600 /root/.ssh/authorized_keys
RUN chmod 700 /root/.ssh/

# install dependencies
RUN pip install transformers==4.26.0 datasets==2.9.0 accelerate==0.16.0 evaluate==0.4.0 google-cloud-storage==2.7.0 google-cloud-aiplatform==1.21.0 tensorboard==2.11.2 tensorflow==2.11.0 tensorflow-io==0.27.0 tensorflow-io-gcs-filesystem==0.27.0
RUN pip install ninja rouge-score nltk  py7zr

#ENV PATH=/usr/local/cuda/bin:$PATH
RUN DS_BUILD_CPU_ADAM=1 \
    DS_BUILD_FUSED_ADAM=1 \
    DS_BUILD_FUSED_LAMB=1 \
    DS_BUILD_UTILS=1 \
    DS_BUILD_AIO=1 \
    pip install deepspeed==0.9.2

# copy deepspeed config file
COPY configs /configs
# copy deepspeed launcher
#COPY run_seq2seq_deepspeed.py run_seq2seq_deepspeed.py
#COPY run_seq2seq_deepspeed-args.py run_seq2seq_deepspeed-args.py
#COPY get_hostfile.py get_hostfile.py

# Sets up the entry point to invoke the trainer with deepspeed
#ENTRYPOINT ["python", "get_hostfile.py"]
#ENTRYPOINT ["deepspeed", "--hostfile=./hostfile", "run_seq2seq_deepspeed-args.py"]

ADD run_seq2seq_deepspeed-args-Copy1.py run.py
ADD start.sh .
CMD ["/bin/sh", "-c", "/start.sh"]
