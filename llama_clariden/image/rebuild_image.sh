#!/bin/bash

# remove sqsh if exists
rm -f megatron.sqsh

podman build -t megatron .
export XDG_RUNTIME_DIR=/dev/shm/$USER/xdg_runtime_dir
mkdir -p $XDG_RUNTIME_DIR

enroot import -o megatron.sqsh podman://megatron

