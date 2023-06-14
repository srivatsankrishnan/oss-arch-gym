#!/usr/bin/env bash
function get_suite {
    URLs=$1
    cat $URLs | while read line
    do
        filename=$(basename $line)
        wget $line
        unzip $filename
        rm $filename
    done
}

function get_simpoints {
    # Get the INT speed SimPoint files.
    mkdir intspeed
    pushd intspeed
    get_suite ../intspeed.urls
    popd

    # Get the FP speed SimPoint files.
    mkdir fpspeed
    pushd fpspeed
    get_suite ../fpspeed.urls
    popd
}

get_simpoints
