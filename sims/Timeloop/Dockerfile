FROM nelliewu/accelergy-timeloop-infrastructure:latest AS builder
FROM timeloopaccelergy/accelergy-timeloop-infrastructure:latest

# Install the base utilities
RUN apt-get update -y \
    && apt-get install -y sudo \
    && apt-get install -y less \
    && apt-get install -y build-essential \
    && apt-get install -y wget \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install miniconda
ENV CONDA_DIR /opt/conda
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p /opt/conda

# Put conda on the path
ENV PATH=$CONDA_DIR/bin:$PATH

RUN cp -r /usr/local/src .
WORKDIR /home/workspace/src
RUN git clone https://github.com/susobhang70/timeloop-examples.git
WORKDIR /home/workspace/src/timeloop-examples/workspace/final-project/example_designs/eyeriss_like
RUN touch /home/workspace/.nointro
RUN chown -R 1000:root /home/workspace && chmod -R 775 /home/workspace

#RUN conda install
RUN git clone --recursive https://github.com/srivatsankrishnan/oss-arch-gym.git
RUN chown -R 1000:root /home/workspace && chmod -R 775 /home/workspace
RUN cd oss-arch-gym && conda env create -f environment.yml

# Make RUN commands use the new environment:
RUN echo "conda activate arch-gym" >> ~/.bashrc
SHELL ["conda", "run", "-n", "arch-gym", "/bin/bash", "--login", "-c"]

WORKDIR /home/workspace/src/timeloop-examples/workspace/final-project/example_designs/eyeriss_like/oss-arch-gym
RUN ./install_sim.sh viz
RUN rm -r ./Project_FARSI
RUN git config --global --add safe.directory /home/workspace/src/timeloop-examples/workspace/final-project/example_designs/eyeriss_like/oss-arch-gym
RUN git config --global --add safe.directory /home/workspace/src/timeloop-examples
RUN git rm -r Project_FARSI
RUN ./install_sim.sh farsi

ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/conda/envs/arch-gym/lib/
RUN echo $LD_LIBRARY_PATH

WORKDIR /home/workspace/src/timeloop-examples/workspace/final-project/example_designs/eyeriss_like/oss-arch-gym/sims/Timeloop
ENTRYPOINT ["conda", "run", "-n", "arch-gym", "python", "/home/workspace/src/timeloop-examples/workspace/final-project/example_designs/eyeriss_like/oss-arch-gym/sims/Timeloop/launch_gcp.py"]

