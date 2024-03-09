#!/bin/bash

# Variables
# SCRIPT_PATH=/wynton/protected/home/fhuanglab/ardademirci/Data_Specific_Epithelial/wynton_code

#! /usr/bin/env bash
#$ -S /bin/bash   # Run in bash
#$ -cwd           # Current working directory
#$ -j y           # Join STDERR and STDOUT
#$ -R yes         # SGE host reservation, highly recommended

conda activate prostate
trap 'conda deactivate' EXIT

# Run the python script
python3 train_combined.py