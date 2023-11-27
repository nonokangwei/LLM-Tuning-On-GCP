# Dockerfile for vLLM serving.
#
# To build:
# docker build -f model_oss/vllm/dockerfile/serve.Dockerfile . -t ${YOUR_IMAGE_TAG}
#
# To push to gcr:
# docker tag ${YOUR_IMAGE_TAG} gcr.io/{YOUR_PROJECT}/${YOUR_IMAGE_TAG}
# docker push gcr.io/{YOUR_PROJECT}/${YOUR_IMAGE_TAG}

# The base image is required by vllm:
# https://vllm.readthedocs.io/en/latest/getting_started/installation.html
# Refer to the nvcr docker hub for the full list:
# https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch/tags
FROM nvcr.io/nvidia/pytorch:22.12-py3

USER root

# Install tools.
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update
RUN apt-get install -y --no-install-recommends apt-utils
RUN apt-get install -y --no-install-recommends curl
RUN apt-get install -y --no-install-recommends wget
RUN apt-get install -y --no-install-recommends git

# Copy license.
RUN wget https://raw.githubusercontent.com/GoogleCloudPlatform/vertex-ai-samples/main/LICENSE

# Install libraries.
ENV PIP_ROOT_USER_ACTION=ignore
RUN python -m pip install --upgrade pip
RUN pip install google-cloud-storage==2.7.0
RUN pip install absl-py==1.4.0

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

# Install vllm deps.
RUN pip install ninja==1.11.1
RUN pip install psutil==5.9.5
RUN pip install ray==2.7.0
RUN pip install sentencepiece==0.1.99
RUN pip install fastapi==0.100.1
RUN pip install uvicorn[standard]==0.23.2
RUN pip install pydantic==1.10.12
RUN pip install --upgrade torch==2.0.1 --index-url https://download.pytorch.org/whl/cu118
RUN pip install xformers==0.0.22
RUN pip install transformers==4.34.0

# Install vllm from source.
RUN git clone https://github.com/vllm-project/vllm.git
WORKDIR vllm
# Pin the version to a fixed git commit on 11/06/2023.
# https://github.com/vllm-project/vllm/tree/1a2bbc930135cd3b94fbff2aafbdf5c568acc8bd
RUN git reset --hard 1a2bbc930135cd3b94fbff2aafbdf5c568acc8bd
# Apply a patch to vllm source:
# 1) For models on Huggingface hub: if the model has multiple bin files, each
#    bin file is downloaded separately and gets deleted after loading to GPU
# 2) For models on GCS bucket: each model bin files is download separately
#    and gets deleted after loading to GPU.
COPY ./vllm.patch /tmp/vllm.patch
RUN git apply /tmp/vllm.patch
RUN pip install -e .

# Install peft to merge lora adapter
RUN pip3 install peft && \
    pip3 install aiohttp

RUN mkdir /root/scripts
COPY merge_peft.py /root/scripts/merge_peft.py
COPY launcher.py /root/scripts/launcher.py

# Expose port 7080 for host serving.
EXPOSE 7080