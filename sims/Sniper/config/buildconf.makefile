# This file is auto-generated, changes made to it will be lost. Please edit makebuildscripts.py instead.

SELF_DIR := $(dir $(word $(words $(MAKEFILE_LIST)),$(MAKEFILE_LIST)))
SNIPER_ROOT ?= $(shell readlink -f $(SELF_DIR)/..)

DR_HOME:=
GRAPHITE_CC:=cc
GRAPHITE_CFLAGS:=-mno-sse4 -mno-sse4.1 -mno-sse4.2 -mno-sse4a -mno-avx -mno-avx2 -I${SNIPER_ROOT}/include 
GRAPHITE_CXX:=g++
GRAPHITE_CXXFLAGS:=-mno-sse4 -mno-sse4.1 -mno-sse4.2 -mno-sse4a -mno-avx -mno-avx2 -I${SNIPER_ROOT}/include 
GRAPHITE_LD:=g++
GRAPHITE_LDFLAGS:=-static -L${SNIPER_ROOT}/lib -pthread 
GRAPHITE_LD_LIBRARY_PATH:=
GRAPHITE_UPCCFLAGS:=-I${SNIPER_ROOT}/include  -link-with='g++ -static -L${SNIPER_ROOT}/lib -pthread'
PIN_HOME:=/root/sniper/pin_kit
SNIPER_CC:=cc
SNIPER_CFLAGS:=-mno-sse4 -mno-sse4.1 -mno-sse4.2 -mno-sse4a -mno-avx -mno-avx2 -I${SNIPER_ROOT}/include 
SNIPER_CXX:=g++
SNIPER_CXXFLAGS:=-mno-sse4 -mno-sse4.1 -mno-sse4.2 -mno-sse4a -mno-avx -mno-avx2 -I${SNIPER_ROOT}/include 
SNIPER_LD:=g++
SNIPER_LDFLAGS:=-static -L${SNIPER_ROOT}/lib -pthread 
SNIPER_LD_LIBRARY_PATH:=
SNIPER_UPCCFLAGS:=-I${SNIPER_ROOT}/include  -link-with='g++ -static -L${SNIPER_ROOT}/lib -pthread'
