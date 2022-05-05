FROM nvidia/cuda:10.1-base

RUN apt-key del 7fa2af80
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64/7fa2af80.pub

RUN apt-get update && apt-get install -y \
    vim wget unzip python3-pip \
    libosmesa6-dev libgl1-mesa-glx libgl1-mesa-dev patchelf libglfw3 build-essential

USER root
RUN python3 -m pip install --upgrade pip

ARG UID
ARG USER
RUN useradd -u $UID --create-home $USER
WORKDIR /home/$USER

# Python packages
USER $USER
RUN mkdir -p /home/$USER/.mujoco \
    && wget https://mujoco.org/download/mujoco210-linux-x86_64.tar.gz -O mujoco.tar.gz \
    && tar -xf mujoco.tar.gz -C /home/$USER/.mujoco \
    && rm mujoco.tar.gz

ENV LD_LIBRARY_PATH /home/$USER/.mujoco/mujoco210/bin:${LD_LIBRARY_PATH}
ENV LD_LIBRARY_PATH /usr/local/nvidia/lib64:${LD_LIBRARY_PATH}

# install packages
RUN python3 -m pip install gym==0.15.4 mujoco-py==2.1.2.14 tensorflow==1.15.5

USER root
COPY ./robosumo ./robosumo
RUN pip install -e ./robosumo

COPY ./baselines ./baselines
RUN python3 -m pip install -e ./baselines

USER $USER

# working dir
WORKDIR /home/$USER/selfplay
CMD /bin/bash