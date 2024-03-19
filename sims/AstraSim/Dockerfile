FROM continuumio/miniconda3
RUN if ! id -u 1000; then useradd -m -u 1000 clouduser; fi
RUN mkdir /workdir
WORKDIR /workdir
RUN echo "recloning arch-gym rep2"
RUN git clone https://github.com/srivatsankrishnan/oss-arch-gym.git

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
WORKDIR /workdir/oss-arch-gym/sims/AstraSim/astrasim_archgym_public/astra-sim/extern/graph_frontend/chakra
RUN git checkout 89f26a5
WORKDIR /workdir

# congestion aware
# congestion aware
RUN cd oss-arch-gym/sims/AstraSim/astrasim_archgym_public/astra-sim/extern/network_backend/analytical && git checkout congestion-multidim
RUN apt-get update && apt-get -y install protobuf-compiler libprotobuf-dev python3-protobuf cmake gcc g++

RUN chmod +x oss-arch-gym/sims/AstraSim/astrasim_archgym_public/astra-sim/build/astra_analytical/build.sh
RUN cd oss-arch-gym/sims/AstraSim/astrasim_archgym_public/astra-sim/ && ./build/astra_analytical/build.sh
RUN chmod +x oss-arch-gym/sims/AstraSim/launch_gcp.py

RUN cd oss-arch-gym && conda env create -f environment.yml
RUN apt-get update && apt-get -y install libgmp-dev gcc g++ libboost-all-dev
RUN echo "conda activate arch-gym" >> ~/.bashrc
SHELL ["/bin/bash", "--login", "-c"]

RUN cd oss-arch-gym/acme && pip install .[orbax-checkpoint,orbax-export]
RUN cd oss-arch-gym/acme && pip install .[jax,tf,testing,envs] && pip install envlogger[tfds] scons && apt-get update && apt-get -y install libgmp-dev && pip install scikit-optimize sympy plotly && conda install --channel conda-forge pygraphviz && conda install -c anaconda python-graphviz && conda install -c anaconda pydot

RUN chown -R 1000:root /workdir && chmod -R 775 /workdir
WORKDIR /workdir/oss-arch-gym/sims/AstraSim

ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "arch-gym", "python", "launch_gcp.py", "--experiment=./experiments_files/experiment.yml", "summary_dir=./all_logs"]
