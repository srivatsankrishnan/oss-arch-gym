# Arch gym - CFU Playground installation

An installation script has been provided to install all dependencies in one go. It's recommended to run this as the superuser, as it asks for your password during execution:

For e.g. use ```sudo bash ./install_sim.sh cfu```, if you want to use bash. You can also set the current terminal to superuser using ```sudo su```, and then run the script normally as ```./install_sim cfu```

If you want to manually install cfu, follow these steps:
- In the oss-arch-gym directory, run 
```sh
git submodule update --init sims/CFU-Playground/CFU-Playground
```

- Move into the CFU Playrgoud directory:
```sh 
cd sims/CFU-Playground/CFU-Playground
```
- Run the following from this location:
```sh
./scripts/setup
./scripts/setup_vexriscv_build.sh
make install-sf
```
Now you should be able to use the CFU-env gym environment.