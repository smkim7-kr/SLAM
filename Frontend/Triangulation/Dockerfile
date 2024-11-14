FROM --platform=linux/amd64 ubuntu:focal

MAINTAINER smkim
ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update -y && apt-get upgrade -y

RUN apt-get install build-essential -y && \
    apt-get install cmake -y && \
    apt-get install git -y && \
    apt-get install sudo -y && \
    apt-get install wget -y && \
    apt-get install ninja-build -y && \
    apt-get install software-properties-common -y && \
    apt-get install python3 -y && \
    apt-get install python3-pip -y && \
    apt-get install -y ssh && \
    apt-get install -y gcc && \
    apt-get install -y g++ && \
    apt-get install -y gdb && \
    apt-get install -y cmake && \
    apt-get install -y rsync && \
    apt-get install -y tar && \
    apt-get install -y x11-utils && \
    apt-get install -y x11-apps && \
    apt-get install -y zip &&\
    apt-get clean

# OpenCV
RUN apt-get install -y cmake git libgtk2.0-dev pkg-config libavcodec-dev libavformat-dev libswscale-dev
RUN pip3 install opencv-python opencv-contrib-python

# Eigen
RUN pip3 install numpy

# Adding visualization dependencies
RUN pip3 install wgpu
RUN pip3 install pygfx  
RUN pip3 install jupyter  
RUN pip3 install ipywidgets  
RUN pip3 install ipygfx  

# Removing Pangolin installation
# Adding Rerun visualizer as an alternative option
RUN pip3 install rerun-sdk

# Replacing PCL with Open3D
RUN pip3 install open3d

# Removing Ceres-solver (not needed for Python implementation)
# For optimization tasks we'll use scipy or similar Python libraries
RUN pip3 install scipy

# Additional Python packages that might be useful
RUN pip3 install matplotlib pandas
