FROM nvcr.io/nvidia/cuda:11.8.0-devel-ubuntu22.04 as base

WORKDIR /

# Install necessary dependencies.
RUN apt update && \
    apt install -y python3-pip python3-packaging \
    wget \
    curl \
    bash \
    vim \
    git \
    ninja-build && \
    pip3 install -U pip

# Install gcloud for access gcs model files
RUN wget -nv \
    https://dl.google.com/dl/cloudsdk/release/google-cloud-sdk.tar.gz && \
    mkdir /root/tools && \
    tar xvzf google-cloud-sdk.tar.gz -C /root/tools && \
    rm google-cloud-sdk.tar.gz && \
    /root/tools/google-cloud-sdk/install.sh --usage-reporting=false \
        --path-update=false --bash-completion=false \
        --disable-installation-options && \
    rm -rf /root/.config/* && \
    ln -s /root/.config /config && \
    # Remove the backup directory that gcloud creates
    rm -rf /root/tools/google-cloud-sdk/.install/.backup

# Gcloud path configuration
ENV PATH $PATH:/root/tools/google-cloud-sdk/bin
# Make sure gsutil will use the default service account
RUN echo '[GoogleCompute]\nservice_account = default' > /etc/boto.cfg

# Tweak this list to reduce build time
# https://developer.nvidia.com/cuda-gpus
ENV TORCH_CUDA_ARCH_LIST "7.0;7.2;7.5;8.0;8.6;8.9;9.0"

# We have to manually install Torch otherwise apex & xformers won't build
RUN pip install torch==2.0.1+cu118 torchvision==0.15.2+cu118 --index-url https://download.pytorch.org/whl/cu118
# To enable H100 PCIe support, install PyTorch >=2.2.0 by uncommenting the following line
# RUN pip3 install "torch==2.2.0.dev20231018+cu118" --index-url https://download.pytorch.org/whl/nightly/cu118

# Install vllm and necessary dependencies
RUN pip3 install "xformers==0.0.22" "transformers==4.34.0" "fschat[model_worker]==0.2.30"

# Install peft to merge lora adapter
RUN pip3 install peft && \
    pip3 install ray && \
    pip3 install aiohttp

# Install google-cloud-storage to download 
RUN pip3 install google-cloud-storage

WORKDIR /root

# download vllm repo to do benchmark test
RUN cd /root && git clone https://github.com/vllm-project/vllm.git
WORKDIR /root/vllm
# Pin the version to a fixed git commit on 11/06/2023.
# https://github.com/vllm-project/vllm/tree/1a2bbc930135cd3b94fbff2aafbdf5c568acc8bd
RUN git reset --hard 1a2bbc930135cd3b94fbff2aafbdf5c568acc8bd
COPY ../vllm.patch /tmp/vllm.patch
RUN git apply /tmp/vllm.patch
RUN pip install -e .

WORKDIR /root

# download ShareGPT_V3_unfiltered_cleaned_split dataset to run benchmark test
RUN mkdir /root/datasets && cd /root/datasets && wget https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json

RUN mkdir /root/scripts
COPY merge_peft.py /root/scripts/merge_peft.py
COPY launcher.py /root/scripts/launcher.py


# ENTRYPOINT ["tail", "-f", "/dev/null"]