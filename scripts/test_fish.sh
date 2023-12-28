DATA=$1
GPU=$2
EXP=$3

dirname="model_ckpt/${DATA}/exp_${EXP}/inference_fish"
mkdir -p -- "$dirname"
python3 -m tools.test_fish --config config_files/fish.yaml \
					 		--gpus ${GPU} --exp ${EXP} --enc one --num_tokens 10 --patch_size 16 --prompt multi --ckpt_used 182_best \
							    | tee -a ${dirname}/log.txt