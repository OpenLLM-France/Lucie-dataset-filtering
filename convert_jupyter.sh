#!/bin/sh

eval "$(conda shell.bash hook)"

conda activate 

python /home/ed/Dev/CODE/Bloomer-dataset-processing/convert_jupyter.py
