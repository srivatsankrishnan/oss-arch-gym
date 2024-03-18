# assuming already ran git clone https://github.com/srivatsankrishnan/oss-arch-gym.git 

#!/bin/bash

cd sims/AstraSim
sudo docker build -t astrasim-archgym .
# echo docker run -v /home/archgym/workspace/aditi_jared/oss-arch-gym/sims/AstraSim/docker_logs/:/workdir/oss-arch-gym/sims/AstraSim/all_logs -it astrasim-archgym