FROM continuumio/miniconda3
RUN if ! id -u 1000; then useradd -m -u 1000 clouduser; fi
RUN mkdir /workdir
WORKDIR /workdir
RUN echo "recloning arch-gym rep223456789"
RUN git clone https://github.com/srivatsankrishnan/oss-arch-gym.git
RUN cd oss-arch-gym && git checkout automate-rw

WORKDIR /workdir/oss-arch-gym
RUN git submodule update --init
WORKDIR /workdir/oss-arch-gym/sims/AstraSim/astrasim_archgym_public/
RUN git submodule update --init
WORKDIR /workdir/oss-arch-gym/sims/AstraSim/astrasim_archgym_public/astra-sim
RUN cp ../../.replace_gitmodules/astrasim.gitmodules .gitmodules && git submodule update --init
WORKDIR /workdir/oss-arch-gym/sims/AstraSim/astrasim_archgym_public/astra-sim/extern/network_backend/analytical
RUN cp ../../../../../.replace_gitmodules/astra-network-analytical.gitmodules .gitmodules && git submodule update --init
WORKDIR /workdir/oss-arch-gym
RUN git submodule update --init --recursive
WORKDIR /workdir

RUN apt-get update && apt-get -y install protobuf-compiler libprotobuf-dev python3-protobuf cmake gcc g++
RUN cd oss-arch-gym/sims/AstraSim/astrasim_archgym_public/astra-sim/ && ./build/astra_analytical/build.sh

RUN cd oss-arch-gym && conda env create -f environment.yml
RUN apt-get update && apt-get -y install libgmp-dev gcc g++ libboost-all-dev
RUN echo "conda activate arch-gym" >> ~/.bashrc
SHELL ["/bin/bash", "--login", "-c"]

RUN cd oss-arch-gym/acme && pip install .[orbax-checkpoint,orbax-export]
RUN cd oss-arch-gym/acme && pip install .[jax,tf,testing,envs] && pip install envlogger[tfds] scons && apt-get update && apt-get -y install libgmp-dev && pip install scikit-optimize sympy plotly && conda install --channel conda-forge pygraphviz && conda install -c anaconda python-graphviz && conda install -c anaconda pydot

RUN chown -R 1000:root /workdir && chmod -R 775 /workdir
WORKDIR /workdir/oss-arch-gym/sims/AstraSim

RUN chmod +x launch_gcp.py

ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "arch-gym", "python", "launch_gcp.py", "--algo=rw"]