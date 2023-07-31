#/bin/sh

#install cfu-playground submodule
if [ $1  == 'cfu' ]; then
    git submodule update --init sims/CFU-Playground/CFU-Playground

    cd sims/CFU-Playground/CFU-Playground

    ./scripts/setup
    ./scripts/setup_vexriscv_build.sh
    make install-sf
fi