FROM nvidia/cuda:11.8.0-runtime-ubuntu20.04

# Installs necessary dependencies.
RUN apt-get update && DEBIAN_FRONTEND=noninteractive TZ=Etc/UTC apt-get install -y --no-install-recommends \
         wget \
         curl \
         git  \
         git-lfs\
         software-properties-common
     
RUN add-apt-repository ppa:deadsnakes/ppa -y
    
RUN apt-get update && \
    apt-get install -y python3.10
    
# RUN apt-get install -y python3-pip && \
#      rm -rf /var/lib/apt/lists/*

# Installs pip.
RUN curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py && \
    python3.10 get-pip.py && \
    pip install setuptools && \
    rm get-pip.py

# Pip
# RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.10
# RUN echo 'export PATH=/home/<USER>/.local/bin:$PATH' >>~/.profile
# RUN source ~/.profile
# RUN pip install --upgrade pip

# install torch-nightly
RUN pip3 install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu118

# install llama-recipes
RUN git clone https://github.com/facebookresearch/llama-recipes.git && \
    pip install --extra-index-url https://download.pytorch.org/whl/test/cu118 -e ./llama-recipes

# Remap
RUN apt-get install python-is-python3
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1

# Installs google cloud sdk, this is mostly for using gsutil to export model.
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

# Path configuration
ENV PATH $PATH:/root/tools/google-cloud-sdk/bin
# Make sure gsutil will use the default service account
RUN echo '[GoogleCompute]\nservice_account = default' > /etc/boto.cfg

# Installs google cloud python dependancy
RUN python3 -m pip install google-cloud-logging

RUN python3 -m pip install tensorboard google-cloud-aiplatform[tensorboard]

WORKDIR /root

# Copies the trainer code 
RUN mkdir /root/trainer

COPY trainer/* /root/trainer/

# Copies updated training_utils python file
RUN mkdir /root/log

# Create a directory to store the model
RUN mkdir -p /root/save/model

COPY train_utils.py /llama-recipes/src/llama_recipes/utils/

# Sets up the entry point to invoke the trainer.
ENTRYPOINT ["python", "trainer/debug.py"]