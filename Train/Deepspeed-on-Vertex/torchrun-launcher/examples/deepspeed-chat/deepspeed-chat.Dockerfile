#
# Copyright 2023 Google LLC
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     https://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
FROM us-docker.pkg.dev/vertex-ai/training/pytorch-gpu.1-13.py310:latest

# Addressing minor quirks in the base image...
ENV PYTHONPATH=/opt/conda/lib/python3.10/site-packages:${PYTHONPATH}
RUN chmod 666 /var/log-storage/output.log

ENV STAGE_DIR=/tmp
RUN mkdir -p ${STAGE_DIR}

RUN apt-get update \
 && apt-get install -y --no-install-recommends \
 openssh-client openssh-server \
 dnsutils iputils-ping \
 net-tools \
 libaio-dev cmake ninja-build pdsh \
 && rm -rf /var/lib/apt/lists/*

########################
# Installing Huggingface and Deepspeed related packages
COPY config/requirements.txt requirements.txt
RUN pip install -r requirements.txt

RUN DS_BUILD_CPU_ADAM=1 DS_BUILD_FUSED_ADAM=1 \
    DS_BUILD_FUSED_LAMB=1 DS_BUILD_TRANSFORMER=1 \
    DS_BUILD_TRANSFORMER_INFERENCE=1 DS_BUILD_STOCHASTIC_TRANSFORMER=1 \
    DS_BUILD_UTILS=1 DS_BUILD_SPARSE_ATTN=0 \
    pip install "deepspeed==0.10.2" --global-option="build_ext"

########################
RUN pip install py7zr

########################
# Application Set up
RUN mkdir -p /root/.cache/huggingface
# You may need a huggingface read access token for certain assets.
COPY token /root/.cache/huggingface/token

# DeepSpeed-Chat
COPY third_party/deepspeed_examples/utils utils
COPY third_party/deepspeed_examples/main.py .
COPY examples/deepspeed-chat/start.sh .
COPY examples/deepspeed-chat/sleep.sh .

ENTRYPOINT ["bash", "./start.sh"]
#ENTRYPOINT ["bash", "./sleep.sh"] # for debugging


