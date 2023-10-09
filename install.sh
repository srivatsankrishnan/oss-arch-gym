#/bin/sh

#install cfu-playground submodule
if [ $1  == 'cfu' ]; then
    git submodule update --init sims/CFU-Playground/CFU-Playground

    cd sims/CFU-Playground/CFU-Playground

    ./scripts/setup_vexriscv_build.sh
    ./scripts/setup
    make install-sf
fi

#install vizier in arch-gym conda environment
#Assumes user is in the arch-gym conda environment

if [ $1 == 'viz' ]; then

    git clone https://github.com/ShvetankPrakash/vizier.git
    cd vizier

    sudo apt-get install -y libprotobuf-dev

    pip install -r requirements.txt --use-deprecated=legacy-resolver
    pip install -e .

    ./build_protos.sh

    pip install -r requirements-algorithms.txt
    pip install -r requirements-benchmarks.txt
fi