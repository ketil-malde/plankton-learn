# Use an official Python runtime as a parent image
FROM tensorflow/tensorflow:1.15.2-gpu-py3

# Set the working directory to 
WORKDIR /project

# User configuration - override with --build-arg
ARG user=myuser
ARG group=mygroup
ARG uid=1000
ARG gid=1000

# Some debs want to interact, even with apt-get install -y, this fixes it
ENV DEBIAN_FRONTEND=noninteractive
ENV HOME=/project

# Install any needed packages from apt
RUN apt-get update && apt-get install -y sudo python3 python3-pip git

# Configure user
RUN groupadd -f -g $gid $user
RUN useradd -u $uid -g $gid $user
RUN usermod -a -G sudo $user
RUN passwd -d $user

RUN mkdir /src
COPY src /src/

# keras 2.4.0 requires tensorflow 2.2, but Matterport Mask R-CNN doesn't support it
# RUN pip3 install --trusted-host pypi.python.org numpy scipy Pillow cython matplotlib scikit-image keras==2.3.1 opencv-python h5py imgaug IPython progressbar2
RUN apt-get install -y python3-pil
RUN pip3 install --trusted-host pypi.python.org keras==2.3.1

# Run when the container launches
CMD "bash"
