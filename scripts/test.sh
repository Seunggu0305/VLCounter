DATA=$1
GPU=$2
EXP=$3

dirname="model_ckpt/${DATA}/exp_${EXP}/inference"
mkdir -p -- "$dirname"
python3 -m tools.test --config config_files/${DATA}.yaml \
					 		--gpus ${GPU} --exp ${EXP} --enc one --num_tokens 10 --patch_size 8 --prompt multi --ckpt_used 200_best \
							    | tee ${dirname}/log.txt