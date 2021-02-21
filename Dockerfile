FROM nvidia/cuda:11.0-cudnn8-runtime-ubuntu18.04
#================================
# Install basics and emboss
#================================
RUN apt-get update -y && apt-get install -y --no-install-recommends \
    python3-dev \
    python3-pip \
    python3-wheel \
    python3-setuptools \
    g++ \
    emboss \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* /var/cache/apt/archives/*

#================================
# Install python library
#================================
RUN pip3 install -U --no-cache-dir pip
RUN pip install --no-cache-dir \
    chainer==7.7.0 \
    cupy-cuda110==7.8.0 \
    pandas==1.1.5 \
    biopython==1.78 \
    prody==2.0

RUN ln -s /usr/bin/python3 /usr/bin/python
RUN mkdir /.cupy && chmod 777 /.cupy

CMD [ "/bin/bash" ]