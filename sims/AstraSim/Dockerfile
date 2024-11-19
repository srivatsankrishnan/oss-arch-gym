FROM continuumio/miniconda3
RUN if ! id -u 1000; then useradd -m -u 1000 clouduser; fi
RUN mkdir /workdir
WORKDIR /workdir
RUN echo "recloning arch-gym rep4fixedBO6-1"
RUN git clone https://github.com/srivatsankrishnan/oss-arch-gym.git

WORKDIR /workdir/oss-arch-gym
RUN git submodule update --init
WORKDIR /workdir/oss-arch-gym/sims/AstraSim/astrasim_archgym_public/
RUN git submodule update --init
WORKDIR /workdir/oss-arch-gym/sims/AstraSim/astrasim_archgym_public/astra-sim
RUN git submodule update --init
WORKDIR /workdir/oss-arch-gym/sims/AstraSim/astrasim_archgym_public/astra-sim/extern/network_backend/analytical
RUN cp ../../../../../.replace_gitmodules/astra-network-analytical.gitmodules .gitmodules && git submodule update --init
WORKDIR /workdir/oss-arch-gym
RUN git submodule update --init --recursive
WORKDIR /workdir

# congestion aware

RUN apt-get update && apt-get -y install protobuf-compiler libprotobuf-dev python3-protobuf cmake gcc g++

RUN chmod +x oss-arch-gym/sims/AstraSim/astrasim_archgym_public/astra-sim/build/astra_analytical/build.sh
RUN cd oss-arch-gym/sims/AstraSim/astrasim_archgym_public/astra-sim/ && ./build/astra_analytical/build.sh
RUN chmod +x oss-arch-gym/sims/AstraSim/launch_gcp.py

RUN cd oss-arch-gym && conda env create -f environment.yml
# update sklearn to 1.2.2
RUN cd oss-arch-gym && pip uninstall scikit-learn && pip install scikit-learn==1.2.2

RUN apt-get update && apt-get -y install libgmp-dev gcc g++ libboost-all-dev
RUN echo "conda activate arch-gym" >> ~/.bashrc
SHELL ["/bin/bash", "--login", "-c"]

RUN cd oss-arch-gym/acme && pip install .[orbax-checkpoint,orbax-export]
# RUN cd oss-arch-gym/acme && pip install .[jax,tf,testing,envs] && pip install envlogger[tfds] scons && apt-get update && apt-get -y install libgmp-dev && pip install scikit-optimize sympy plotly && conda install --channel conda-forge pygraphviz && conda install -c anaconda python-graphviz && conda install -c anaconda pydot
# RUN cd oss-arch-gym/acme && pip install .[orbax-checkpoint==0.9.1,orbax-export==0.1.0]

# Install 'jax', 'tf', 'testing', and 'envs' extras for 'acme' with specified versions
RUN cd oss-arch-gym/acme && pip install '.[testing,envs]' \
    && pip install 'jax==0.3.6' 'tensorflow==2.8.0' \
    && pip install 'envlogger[tfds]==1.0.8' 'scons==4.5.2' \
    && sudo apt-get update && sudo apt-get -y install libgmp-dev \
    && pip install 'scikit-optimize==0.9.0' 'sympy==1.12' 'plotly==5.18.0' \
    && conda install --channel conda-forge 'pygraphviz==1.11' \
    && conda install -c anaconda 'python-graphviz==0.20.1' \
    && conda install -c anaconda 'pydot==1.4.2'


RUN chown -R 1000:root /workdir && chmod -R 775 /workdir
WORKDIR /workdir/oss-arch-gym/sims/AstraSim

ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "arch-gym", "python", "launch_gcp.py", "--experiment=./11-17-2024-new_experiment-commands/experiment8a_gpt3_175b.yml", "--summary_dir=./11-17-2024-new_experiment-logs/experiment8a_gpt3_175b_log", "--timeout=604800"]
