FROM nvcr.io/nvidia/tensorrt:25.05-py3

RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y

WORKDIR /work

ENTRYPOINT ["/bin/bash"]