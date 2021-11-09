# Copyright 2019 The MediaPipe Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

FROM nvidia/cuda:11.2.0-cudnn8-devel-ubuntu18.04

ARG OPENCV_VERSION=4.5.3

LABEL maintainer <mediapipe@google.com>

WORKDIR /io
WORKDIR /mediapipe

ENV DEBIAN_FRONTEND=noninteractive

RUN apt update
RUN apt install software-properties-common -y
RUN add-apt-repository ppa:deadsnakes/ppa

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        cmake \
        yasm \
        libtbb2 \
        libpng-dev \
        libpq-dev \
        libxine2-dev \
        libglew-dev \
        libtiff5-dev \
        zlib1g-dev \
        libavutil-dev \
        libpostproc-dev \
        libeigen3-dev \
        ## Python
        python3-numpy \
        gcc-8 g++-8 \
        ca-certificates \
        curl \
        ffmpeg \
        git \
        vim \
        wget \
        unzip \
        python3.7-dev \
        python3-pip \
        libopencv-core-dev \
        libopencv-highgui-dev \
        libopencv-imgproc-dev \
        libopencv-video-dev \
        libopencv-calib3d-dev \
        libopencv-features2d-dev \
        software-properties-common \
        python3.7-venv libprotobuf-dev protobuf-compiler cmake libgtk2.0-dev \
        mesa-common-dev libegl1-mesa-dev libgles2-mesa-dev mesa-utils \
        pkg-config libgtk-3-dev libavcodec-dev libavformat-dev libswscale-dev libv4l-dev \
        libxvidcore-dev libx264-dev libjpeg-dev libpng-dev libtiff-dev \
        gfortran openexr libatlas-base-dev libhdf5-dev\
        libtbb2 libtbb-dev libdc1394-22-dev ffmpeg libgles2 libegl1-mesa-dev && \
    add-apt-repository -y ppa:openjdk-r/ppa && \
    apt-get update && apt-get install -y openjdk-8-jdk && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-8 100 --slave /usr/bin/g++ g++ /usr/bin/g++-8
RUN python3.7 -m pip install --upgrade pip
RUN python3.7 -m pip install --upgrade setuptools
RUN python3.7 -m pip install wheel
RUN python3.7 -m pip install future
RUN python3.7 -m pip install numpy
RUN python3.7 -m pip install six
RUN python3.7 -m pip install tensorflow-gpu
# RUN python3.7 -m pip install opencv-python
# RUN python3.7 -m pip install opencv-contrib-python
RUN python3.7 -m pip install install tf_slim

RUN ln -s /usr/bin/python3.7 /usr/bin/python

# Build opencv
RUN cd /opt/ &&\
    # Download and unzip OpenCV and opencv_contrib and delte zip files
    wget https://github.com/opencv/opencv/archive/$OPENCV_VERSION.zip &&\
    unzip $OPENCV_VERSION.zip &&\
    rm $OPENCV_VERSION.zip &&\
    wget https://github.com/opencv/opencv_contrib/archive/$OPENCV_VERSION.zip &&\
    unzip ${OPENCV_VERSION}.zip &&\
    rm ${OPENCV_VERSION}.zip &&\
    # Create build folder and switch to it
    mkdir /opt/opencv-${OPENCV_VERSION}/build && cd /opt/opencv-${OPENCV_VERSION}/build &&\
    # Cmake configure
    cmake \
        -DOPENCV_EXTRA_MODULES_PATH=/opt/opencv_contrib-${OPENCV_VERSION}/modules \
        -DWITH_CUDA=ON \
        -DCMAKE_BUILD_TYPE=RELEASE \
        # Install path will be /usr/local/lib (lib is implicit)
        -DCMAKE_INSTALL_PREFIX=/usr/local \
        .. &&\
    # Make
    make -j"$(nproc)" && \
    # Install to /usr/local/lib
    make install && \
    ldconfig &&\
    # Remove OpenCV sources and build folder
    rm -rf /opt/opencv-${OPENCV_VERSION} && rm -rf /opt/opencv_contrib-${OPENCV_VERSION}


# Install bazel
ARG BAZEL_VERSION=3.7.2
RUN mkdir /bazel && \
    wget --no-check-certificate -O /bazel/installer.sh "https://github.com/bazelbuild/bazel/releases/download/${BAZEL_VERSION}/b\
azel-${BAZEL_VERSION}-installer-linux-x86_64.sh" && \
    wget --no-check-certificate -O  /bazel/LICENSE.txt "https://raw.githubusercontent.com/bazelbuild/bazel/master/LICENSE" && \
    chmod +x /bazel/installer.sh && \
    /bazel/installer.sh  && \
    rm -f /bazel/installer.sh

COPY . /mediapipe/

RUN python -m pip install -r requirements.txt

# If we want the docker image to contain the pre-built object_detection_offline_demo binary, do the following
# RUN bazel build -c opt --define MEDIAPIPE_DISABLE_GPU=1 mediapipe/examples/desktop/demo:object_detection_tensorflow_demo


ENV PATH=/usr/local/cuda-11.2/bin${PATH:+:${PATH}}
ENV LD_LIBRARY_PATH=/usr/local/cuda/extras/CUPTI/lib64,/usr/local/cuda-11.2/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
RUN ldconfig

ENV TF_CUDA_PATHS=/usr/local/cuda-11.2,/usr/lib/x86_64-linux-gnu,/usr/include
# ENV TF_CUDA_PATHS=/usr/local/cuda-10.1,/usr/lib/x86_64-linux-gnu,/usr/include

COPY tests /tmp

# RUN cp /usr/include/linux/cuda.h /usr/include/
RUN python setup.py gen_protos
# RUN python setup.py bdist_wheel
RUN python setup.py install --link-opencv
# RUN python -m pip install -e .


# RUN python /mediapipe/mediapipe/python/solutions/pose_test.py


