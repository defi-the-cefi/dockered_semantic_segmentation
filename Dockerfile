# MAKEFILE
FROM tensorflow/tensorflow:latest-gpu

ARG DEBIAN_FRONTEND=noninteractive

RUN rm /etc/apt/sources.list.d/cuda.list
RUN rm /etc/apt/sources.list.d/nvidia-ml.list
RUN apt-key del 7fa2af80
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/3bf863cc.pub
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu2004/x86_64/7fa2af80.pub

RUN apt-get update && apt-get install -y git wget vim python3-venv
        #&& git clone https://github.com/defi-the-cefi/semantic_segmentation.git

RUN mkdir ./semantic_segmentation

# https://stackoverflow.com/questions/27701930/how-to-add-users-to-docker-container
RUN groupadd --gid 1200 guest \
    && useradd --uid 1200 --gid guest --shell /bin/bash --create-home guest

RUN chown -R guest:guest ./semantic_segmentation

# RUN useradd --user-group guest --create-home --no-log-init guest
# --system --shell /bin/bash
USER guest

WORKDIR "./semantic_segmentation"
RUN pwd

# copy files from gitcloned repo into our container
ADD "./images" "./images"
ADD "./output_images" "./output_images"
ADD "./seg_model/." "./"
RUN ls

RUN python3 -m venv ./segv
ENV PATH="./segv/bin:$PATH"
RUN pip3 install torch torchvision torchaudio
RUN mkdir ./.torch_cache
ENV TORCH_HOME="./.torch_cache"

ENTRYPOINT python3 seg_model.py
