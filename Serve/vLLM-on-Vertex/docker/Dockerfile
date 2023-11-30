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
FROM us-docker.pkg.dev/vertex-ai/vertex-vision-model-garden-dockers/pytorch-vllm-serve

USER root

# Install tools.
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update

ENV PIP_ROOT_USER_ACTION=ignore

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
RUN pip install fschat[model_worker]==0.2.30

# Install peft to merge lora adapter
RUN pip3 install peft && \
    pip3 install aiohttp

RUN mkdir /root/scripts
COPY merge_peft.py /root/scripts/merge_peft.py
COPY launcher.py /root/scripts/launcher.py

# Expose port 7080 for host serving.
# EXPOSE 7080