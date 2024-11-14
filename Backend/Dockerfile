# Use Ubuntu 22.04 as the base image
FROM ubuntu:22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    git \
    wget \
    curl \
    libeigen3-dev \
    libgoogle-glog-dev \
    libsuitesparse-dev \
    libboost-all-dev \
    python3 \
    python3-pip \
    python3-dev \
    python3-numpy \
    python3-sympy \
    python3-lark-parser \
    python3-yaml \
    ipython3 \
    && rm -rf /var/lib/apt/lists/*

# Set default python to python3
RUN ln -s /usr/bin/python3 /usr/bin/python

# Upgrade pip
RUN pip3 install --upgrade pip

# Install symengine for performance improvements in symforce
RUN pip3 install symengine

# Install symforce
RUN pip3 install symforce

# Install ceres-solver
RUN git clone https://github.com/ceres-solver/ceres-solver.git /tmp/ceres-solver && \
    cd /tmp/ceres-solver && \
    mkdir build && cd build && \
    cmake .. -DBUILD_TESTING=OFF -DBUILD_EXAMPLES=OFF && \
    make -j$(nproc) && \
    make install && \
    cd && rm -rf /tmp/ceres-solver

# Install g2o
RUN git clone https://github.com/RainerKuemmerle/g2o.git /tmp/g2o && \
    cd /tmp/g2o && \
    mkdir build && cd build && \
    cmake .. -DBUILD_SHARED_LIBS=ON -DG2O_BUILD_EXAMPLES=OFF && \
    make -j$(nproc) && \
    make install && \
    cd && rm -rf /tmp/g2o

# Install GTSAM
RUN git clone https://github.com/borglab/gtsam.git /tmp/gtsam && \
    cd /tmp/gtsam && \
    mkdir build && cd build && \
    cmake .. -DGTSAM_BUILD_WITH_MARCH_NATIVE=OFF \
             -DGTSAM_BUILD_EXAMPLES_ALWAYS=OFF \
             -DGTSAM_BUILD_TESTS=OFF \
             -DGTSAM_USE_SYSTEM_EIGEN=ON && \
    make -j$(nproc) install && \
    cd && rm -rf /tmp/gtsam
