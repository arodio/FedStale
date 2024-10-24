#!/bin/bash

for seed in 42 78 84;do
	echo $seed && oarsub -p "gpu='YES' and host='nefgpu52.inria.fr'" -l /gpunum=1,walltime=4 -t idempotent "./server_run.sh $seed"
done
