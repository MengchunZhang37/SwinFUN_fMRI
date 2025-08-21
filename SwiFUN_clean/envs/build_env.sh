#!/bin/bash
#SBATCH --job-name=buildenv
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=buildenv.out
#SBATCH --error=buildenv.err

source ~/.bashrc

conda env remove -n py39 -y

conda env create -f /ihome/haizenstein/mez141/ondemand/py39.yaml

source activate py39  # 有些系统需要 conda activate py39

pip install --no-cache-dir --user -r /ihome/haizenstein/mez141/ondemand/requirements.txt


