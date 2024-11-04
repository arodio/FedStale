#!/bin/bash

for seed in 42 78 84;do
	for lr in "1e-2" "5e-2" "1e-3";do
		echo $seed $lr && oarsub -p "gpu='YES' and host='nefgpu52.inria.fr'" -l /gpunum=1,walltime=55 -t idempotent "./server_run.sh $seed 128"
	done
done

for seed in 42 78 84;do
	for lr in "1e-2" "5e-2" "1e-3";do
		echo $seed $lr && oarsub -p "gpu='YES' and host='nefgpu52.inria.fr'" -l /gpunum=1,walltime=55 -t idempotent "./server_run.sh $seed 128"
	done
done
