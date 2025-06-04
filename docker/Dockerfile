FROM pytorch/pytorch:2.7.0-cuda12.8-cudnn9-devel

RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y

WORKDIR /work

ENTRYPOINT ["/bin/bash"]
