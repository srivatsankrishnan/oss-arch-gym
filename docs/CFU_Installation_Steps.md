# Arch gym - CFU Playground installation

We suggest using the installation script : ```./install_sim.sh cfu```, or you can follow the following steps:

- In the oss-arch-gym directory, run 
```sh
git submodule update --init sims/CFU-Playground
```

- Move into the CFU Playrgoud directory:
```sh 
cd sims/CFU-Playground
```
- Run the following from this location:
```sh
./scripts/setup
./scripts/setup_vexriscv_build.sh
make install-sf
```
Now you should be able to use the CFU-env gym environment.