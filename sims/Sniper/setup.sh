#!/usr/bin/env bash
function grab_benchmarks {
    pushd CPU2017
    ./setup.sh
    popd
}

function build_docker_image {
    pushd docker
    ./build.sh
    popd
}

function copy_sniper_configs {
    docker create --name temporary_sniper_container_for_config sniper
    docker cp temporary_sniper_container_for_config:/root/sniper/config .
    docker rm temporary_sniper_container_for_config
}

grab_benchmarks
build_docker_image
copy_sniper_configs
