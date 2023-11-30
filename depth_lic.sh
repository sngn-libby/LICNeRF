#!/bin/bash

echo "### START DATE=$(date)"
echo "### HOSTNAME=$(hostname)"
echo "### CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

source ~/.bashrc

conda activate lic_env


ml purge
#conda install --file lic_requirements_nover_part.txt
#pip install numpy

#python3 ~/cucuda_s/lic_experiment/LICNeRF/src/data/data_util/nerf_360_v2.py

#python3 run_de.py --model mbt_de --dataset ./configs/licnerf/vimeo90k.gin

python3 run_de.py --seed 777 --model mbt-de --dataset ~/vimeo90k_compressai -td ~/kodak_dataset

echo "###"
echo "### END DATE=$(date)"
