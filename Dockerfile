FROM ubuntu:16.04

MAINTAINER Brian Low <brian.low22@gmail.com>

RUN apt-get update && apt-get install -y \
  nano \
  wget \
  git \
  libopenblas-dev \
  python-dev \
  python-pip \
  python-nose \
  python-numpy \
  python-scipy \
  python-yaml \
  pandoc

RUN pip install --upgrade pip six twine pypandoc

RUN pip install --upgrade --no-deps git+git://github.com/Theano/Theano.git@74c0a4c76610a6571451e0b2094e637ad96cee0b
RUN pip install --upgrade --no-deps git+git://github.com/fchollet/keras.git@1.0.6

COPY setup.py .pypirc README.md /keras_diagram/
COPY keras_diagram/ /keras_diagram/keras_diagram/

WORKDIR /keras_diagram



