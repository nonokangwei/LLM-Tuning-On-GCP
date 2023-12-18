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
RUN pip install datasets==2.9.0 sentencepiece==0.1.97 protobuf==3.20.3 accelerate==0.20.3 transformers==4.31.0 tensorboard
RUN pip install ninja rouge-score nltk  py7zr google-cloud google-cloud-aiplatform fsspec==2023.9.2

#ENV PATH=/usr/local/cuda/bin:$PATH
RUN DS_BUILD_CPU_ADAM=1 \
    DS_BUILD_FUSED_ADAM=1 \
    DS_BUILD_FUSED_LAMB=1 \
    DS_BUILD_UTILS=1 \
    DS_BUILD_AIO=1 \
    pip install deepspeed==0.10.2

RUN mkdir -p .cache/huggingface
# You may need a huggingface read access token for certain assets.
ADD token /root/.cache/huggingface/token

# DeepSpeed-Chat
ADD third_party/deepspeed_examples/utils utils
ADD third_party/deepspeed_examples/main.py .

ADD start.sh .
CMD ["/bin/sh", "-c", "/start.sh"]
