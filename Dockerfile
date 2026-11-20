FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    python3.9 python3.9-dev python3-pip \
    git wget build-essential \
    libglib2.0-0 libsm6 libxext6 libxrender-dev \
    libgl1-mesa-glx zlib1g-dev libjpeg-dev \
    vim \
    && rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3.9 /usr/bin/python && \
    python -m pip install --upgrade pip

WORKDIR /oed

RUN mkdir -p /.local /.cache /.config && \
    chmod -R 777 /.local /.cache /.config

RUN pip install torch==2.0.1 torchvision==0.15.2 \
    --index-url https://download.pytorch.org/whl/cu118

COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

COPY . /oed

RUN chmod -R 777 /oed

RUN find /oed -name "*.py" | xargs sed -i \
    -e 's/dtype=np\.int\b/dtype=np.int64/g' \
    -e 's/dtype=np\.float\b/dtype=np.float64/g' \
    -e 's/dtype=np\.bool\b/dtype=bool/g' \
    -e 's/dtype=np\.complex\b/dtype=np.complex128/g'

CMD ["bash"]