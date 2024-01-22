FROM continuumio/miniconda3
RUN if ! id -u 1000; then useradd -m -u 1000 clouduser; fi
RUN mkdir /workdir
WORKDIR /workdir
RUN echo "recloning arch-gym rep22"
#RUN git clone --recursive https://srivatsankrishnan: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@github.com/srivatsankrishnan/arch-gym.git
RUN git clone --recursive https://github.com/srivatsankrishnan/oss-arch-gym.git
# RUN cd oss-arch-gym && git checkout automate-rw
# RUN cd oss-arch-gym && git submodule update --init --recursive
# RUN cd oss-arch-gym && git checkout automate-rw && git submodule update --init --recursive
# RUN cd oss-arch-gym/sims/AstraSim/astrasim_archgym_public/astra-sim/ && ./build/astra_analytical/build.sh

# RUN cd oss-arch-gym
RUN cd oss-arch-gym && conda env create -f environment.yml
RUN apt-get update && apt-get -y install libgmp-dev gcc g++ libboost-all-dev
RUN echo "conda activate arch-gym" >> ~/.bashrc
SHELL ["/bin/bash", "--login", "-c"]

RUN cd oss-arch-gym/acme && pip install .[orbax-checkpoint,orbax-export]
RUN cd oss-arch-gym/acme && pip install .[jax,tf,testing,envs] && pip install envlogger[tfds] scons && apt-get update && apt-get -y install libgmp-dev && pip install scikit-optimize sympy plotly && conda install --channel conda-forge pygraphviz
# WORKDIR /workdir/oss-arch-gym/acme
# RUN pip install jax
# RUN pip install tf
# # RUN pip install testing
# RUN pip install envs
# # RUN pip install .[jax,tf,testing,envs]
# RUN pip install envlogger[tfds] scons && apt-get update && apt-get -y install libgmp-dev && pip install scikit-optimize sympy plotly && conda install --channel conda-forge pygraphviz
# WORKDIR /workdir

RUN chown -R 1000:root /workdir && chmod -R 775 /workdir
WORKDIR /workdir/oss-arch-gym/sims/AstraSim

# ENTRYPOINT ["conda", "run", "-n", "arch-gym"]
