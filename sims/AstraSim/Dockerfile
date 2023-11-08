# Dockerfile for Arch-Gym project for other agents (BO, ACO, GA, RW)
# start with miniconda image


FROM continuumio/miniconda3
RUN if ! id 1000; then useradd -m -u 1000 clouduser; fi
RUN mkdir /workdir
WORKDIR /workdir
RUN echo "recloning arch-gym rep00000o0"
#RUN git clone --recursive https://srivatsankrishnan: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@github.com/srivatsankrishnan/arch-gym.git
RUN git clone --recursive https://srivatsankrishnan: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@github.com/srivatsankrishnan/arch-gym.git

RUN cd arch-gym && conda env create -f environment.yml
RUN apt-get update && apt-get -y install libgmp-dev gcc g++ libboost-all-dev
RUN echo "conda activate arch-gym" >> ~/.bashrc
SHELL ["/bin/bash", "--login", "-c"]


RUN cd arch-gym/acme && pip install .[jax,tf,testing,envs] && pip install envlogger[tfds] scons && apt-get update && apt-get -y install libgmp-dev && pip install scikit-optimize sympy plotly && conda install --channel conda-forge pygraphviz


# The code to run when container is started:


RUN chown -R 1000:root /workdir && chmod -R 775 /workdir


WORKDIR /workdir/arch-gym/sims/AstraSim/


# Install cost model
RUN python build.py

ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/conda/envs/arch-gym/lib
RUN echo $LD_LIBRARY_PATH
WORKDIR /workdir/arch-gym/sims/AstraSim/

ENTRYPOINT ["conda", "run", "-n", "arch-gym", "python", "launch_gcp.py"]