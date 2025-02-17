#!/bin/bash


export XDG_RUNTIME_DIR=/dev/shm/$USER/xdg_runtime_dir
mkdir -p $XDG_RUNTIME_DIR

podman build -t megatron .

enroot import -o megatron_new.sqsh podman://megatron

# remove sqsh if exists
rm -f megatron.sqsh

mv megatron_new.sqsh megatron.sqsh
