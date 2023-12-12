FROM nelliewu/accelergy-timeloop-infrastructure:latest AS builder
#
# Main image
#
FROM timeloopaccelergy/accelergy-timeloop-infrastructure:latest

RUN if ! id 1000; then useradd -m -u 1000 clouduser; fi

# Install the base utilities
RUN apt-get update -y \
    && apt-get install -y less \
    && apt-get install -y build-essential \
    && apt-get install -y wget \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install miniconda
ENV conda /opt/conda
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
RUN apt-get update && apt-get -y install libgmp-dev gcc g++
  
# Make RUN commands use the new environment:
RUN echo "conda activate arch-gym" >> ~/.bashrc
SHELL ["conda", "run", "-n", "arch-gym", "/bin/bash", "--login", "-c"]

RUN cd oss-arch-gym/acme && pip install .[jax,tf,testing,envs] && pip install envlogger[tfds] && cd ../ && pip install -e . && pip install absl-py && pip install scikit-optimize
ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/bin/conda/envs/arch-gym/lib
RUN echo $LD_LIBRARY_PATH

ENTRYPOINT ["conda", "run", "-n", "arch-gym", "python", "/home/workspace/src/timeloop-examples/workspace/final-project/example_designs/eyeriss_like/arch-gym/sims/Timeloop/launch_gcp.py"]

