FROM gcr.io/tensorflow/tensorflow:0.12.1-devel-gpu

# Install system dependancy
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    pkg-config \
    python-tk \

    libavcodec-dev \
    libavformat-dev \
    libswscale-dev\
    libtbb2 \
    libtbb-dev \
    libjpeg-dev \
    libpng-dev \
    libtiff-dev \
    libjasper-dev \
    libdc1394-22-dev

# Download and install opencv
WORKDIR /opt
RUN git clone https://github.com/Itseez/opencv.git && \
    cd opencv && \
    git checkout master && \
    cd /opt/opencv && \
    mkdir release && \
    cd release && \
    cmake -D CMAKE_BUILD_TYPE=RELEASE -D CMAKE_INSTALL_PREFIX=/usr/local \
          -D INSTALL_PYTHON_EXAMPLES=ON /opt/opencv/ && \
    make -j"$(nproc)" && \
    make install && \
    cp /opt/opencv/release/lib/cv2.so /usr/local/lib/python2.7/dist-packages && \
    rm -rf /opt/opencv && \
    ldconfig

WORKDIR /root/workspace
RUN git clone https://github.com/tensorflow/models/

# install python packages
WORKDIR /tmp
ADD requirements.txt .
RUN pip install -r requirements.txt

WORKDIR /root/workspace
