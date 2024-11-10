FROM ubuntu:20.04

RUN apt-get update && \
    apt-get install -y wget bzip2 && \
    apt-get clean

RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh && \
    bash miniconda.sh -b -p /opt/conda && \
    rm miniconda.sh

ENV PATH="/opt/conda/bin:$PATH"
RUN conda init bash

RUN conda create -n signal_p python=3.8 numpy scipy matplotlib tqdm -y
SHELL ["conda", "run", "-n", "signal_p", "/bin/bash", "-c"]
WORKDIR /app
COPY . /app

RUN conda run -n signal_p pip install -r requirements.txt

CMD ["conda", "run", "-n", "signal_p", "python", "main.py"]