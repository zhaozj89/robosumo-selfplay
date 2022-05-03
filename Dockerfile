FROM nvidia/cuda:10.1-base

RUN apt-key del 7fa2af80
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64/7fa2af80.pub

RUN apt-get update && apt-get install -y \
    vim wget unzip \
    libosmesa6-dev libgl1-mesa-glx libgl1-mesa-dev patchelf libglfw3 build-essential

RUN useradd --create-home user
USER user
WORKDIR /home/user

RUN wget -q https://repo.anaconda.com/miniconda/Miniconda3-4.5.4-Linux-x86_64.sh && \
    bash Miniconda3-4.5.4-Linux-x86_64.sh -b -p miniconda3 && \
    rm Miniconda3-4.5.4-Linux-x86_64.sh

ENV PATH /home/user/miniconda3/bin:$PATH

# Python packages
RUN conda install python=3.6

RUN mkdir -p /home/user/.mujoco \
    && wget https://mujoco.org/download/mujoco210-linux-x86_64.tar.gz -O mujoco.tar.gz \
    && tar -xf mujoco.tar.gz -C /home/user/.mujoco \
    && rm mujoco.tar.gz

ENV LD_LIBRARY_PATH /home/user/.mujoco/mujoco210/bin:${LD_LIBRARY_PATH}
ENV LD_LIBRARY_PATH /usr/local/nvidia/lib64:${LD_LIBRARY_PATH}

# install packages
RUN pip install gym==0.15.4 mujoco-py==2.1.2.14 tensorflow==1.15.5
