#!/bin/bash

# Variables
# SCRIPT_PATH=/wynton/protected/home/hong/ardademirci/ecog_script

#! /usr/bin/env bash
#$ -S /bin/bash   # Run in bash
#$ -cwd           # Current working directory
#$ -j y           # Join STDERR and STDOUT
#$ -R yes         # SGE host reservation, highly recommended

conda activate bias
trap 'conda deactivate' EXIT

# Run the python script
python3 main.py --train_population 1