FROM ubuntu:20.04

RUN apt-get update && \
    apt-get install -y wget bzip2 && \
    apt-get clean

RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh && \
    bash miniconda.sh -b -p /opt/conda && \
    rm miniconda.sh

ENV PATH="/opt/conda/bin:$PATH"
RUN conda init bash

RUN conda create -n myenv python=3.8 numpy scipy matplotlib tqdm -y
SHELL ["conda", "run", "-n", "myenv", "/bin/bash", "-c"]
WORKDIR /app
COPY . /app

RUN conda run -n myenv pip install -r requirements.txt

ENV OUTPUT_DIR=/app/Docker_to_PNG

CMD ["conda", "run", "-n", "myenv", "python", "main.py"]

COPY ./figure Docker_to_PNG
