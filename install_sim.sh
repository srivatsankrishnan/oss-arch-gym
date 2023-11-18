#/bin/sh

#install cfu-playground submodule
if [ $1  == 'cfu' ]; then
    git submodule update --init sims/CFU-Playground/CFU-Playground

    cd sims/CFU-Playground/CFU-Playground
    sudo apt-get install -y ninja-build
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

#install Project_FARSI submodule

if [ $1  == 'farsi' ]; then

    # first, delete the current Project_FARSI folder if any

    git submodule add https://github.com/facebookresearch/Project_FARSI.git Project_FARSI

    git submodule update --init Project_FARSI

    echo "---Submodule Project_FARSI has been created!---"

    cd Project_FARSI && git clone https://github.com/zaddan/cacti_for_FARSI.git && cd ..

    echo "---Downloaded cacti_for_FARSI---"
    

    #Use the environment_FARSI.yml file to update the conda env dependencies
    echo "---Updating the conda env with additional dependencies for FARSI---"
    cd sims/FARSI_sim && conda env update -f environment_FARSI.yml && cd ../..

    sudo apt update && sudo apt install -y build-essential && cd Project_FARSI/cacti_for_FARSI && make clean && make

    cd ../..

    echo "---INSTALLING ACME: ---"

    cd acme && pip install .[jax,tf,testing,envs] && cd ..

    echo "---ACME installation completed!---"

    sudo apt-get update && sudo apt-get -y install libgmp-dev && pip install scikit-optimize

fi
