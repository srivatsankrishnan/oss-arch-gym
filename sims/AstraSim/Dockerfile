FROM continuumio/miniconda3
RUN if ! id -u 1000; then useradd -m -u 1000 clouduser; fi
RUN mkdir /workdir
WORKDIR /workdir
RUN echo "recloning arch-gym rep4fixedBO67589-19789548"
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
RUN cd oss-arch-gym/acme && pip install .[jax,tf,testing,envs] && pip install envlogger[tfds] scons && apt-get update && apt-get -y install libgmp-dev && pip install scikit-optimize sympy plotly && conda install --channel conda-forge pygraphviz
RUN conda install -c anaconda python-graphviz
RUN conda install -c anaconda pydot

RUN chown -R 1000:root /workdir && chmod -R 775 /workdir
WORKDIR /workdir/oss-arch-gym/sims/AstraSim

ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "arch-gym", "python", "launch_gcp.py", "--experiment=./11-17-2024-new_experiment-commands/experiment8a_gpt3_175b.yml", "--summary_dir=./11-17-2024-new_experiment-logs/experiment8a_gpt3_175b_log", "--timeout=604800"]
