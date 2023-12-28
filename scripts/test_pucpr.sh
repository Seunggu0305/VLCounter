DATA=$1
GPU=$2
EXP=$3

dirname="model_ckpt/${DATA}/exp_${EXP}/inference_pucpr"
mkdir -p -- "$dirname"
python3 -m tools.test_pucpr --config config_files/${DATA}.yaml \
					 		--gpus ${GPU} --exp ${EXP} --enc one --prompt multi --ckpt_used 182_best \
							    | tee -a ${dirname}/log.txt