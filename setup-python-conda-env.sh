#!/bin/bash

set -e

trap "echo ERROR" ERR

cd "$(dirname "$0")" || exit 1

eval "$(command conda 'shell.bash' 'hook' 2> /dev/null)"

conda deactivate

conda env create -f environment.yml --force

conda activate cherokee-wavernn

pip install -e WaveRNN

exit 0
