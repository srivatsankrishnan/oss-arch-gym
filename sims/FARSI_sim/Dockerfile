# Dockerfile for Arch-Gym project for other agents (BO, ACO, GA, RW)
# start with miniconda image

FROM continuumio/miniconda3
RUN if ! id 1000; then useradd -m -u 1000 clouduser; fi
RUN mkdir /workdir
WORKDIR /workdir
RUN echo "recloing arch-gym rep0000ooooo"
RUN git clone --recursive https://srivatsankrishnan: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@github.com/srivatsankrishnan/arch-gym.git

RUN cd arch-gym && conda env create -f environment.yml
RUN cd arch-gym/sims/FARSI_sim && conda env update -f environment_FARSI.yml
RUN apt update && apt install -y build-essential && cd arch-gym/Project_FARSI/cacti_for_FARSI && make clean && make
# Make RUN commands use the new environment:
RUN echo "conda activate arch-gym" >> ~/.bashrc
SHELL ["/bin/bash", "--login", "-c"]

RUN cd arch-gym/acme && pip install .[jax,tf,testing,envs] && pip install envlogger[tfds] && apt-get update && apt-get -y install libgmp-dev && pip install scikit-optimize

# The code to run when container is started:

RUN chown -R 1000:root /workdir && chmod -R 775 /workdir
WORKDIR /workdir/arch-gym/sims/FARSI_sim
ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/conda/envs/arch-gym/lib
RUN echo $LD_LIBRARY_PATH
ENTRYPOINT ["conda", "run", "-n", "arch-gym", "python", "launch_gcp.py"]