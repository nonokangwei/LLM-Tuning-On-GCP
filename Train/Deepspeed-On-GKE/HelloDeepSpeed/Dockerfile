FROM us-docker.pkg.dev/flius-vpc-2/flius-vpc-2-repo/deepspeed:ubuntu18.04-base

### ssh key ###
RUN sudo service ssh start
ADD id_rsa /home/deepspeed/.ssh/id_rsa
ADD id_rsa.pub /home/deepspeed/.ssh/id_rsa.pub
ADD authorized_keys /home/deepspeed/.ssh/authorized_keys

#USER deepspeed
### Deepspeed examples ###
WORKDIR /tmp
RUN git clone https://github.com/microsoft/DeepSpeedExamples.git
RUN cd DeepSpeedExamples/training/HelloDeepSpeed && \
    sudo apt-get update && \
    sudo apt install -y python-pip && \
    pip install -r requirements.txt

ADD start.sh .
CMD ["/bin/sh", "-c", "/tmp/start.sh"]
