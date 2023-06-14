#!/usr/bin/env bash
SNIPER_LINK=$(cat SniperLink)
docker build -t sniper --build-arg sniper_git=${SNIPER_LINK} -f Dockerfile .
