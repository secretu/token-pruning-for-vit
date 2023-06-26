#!/bin/bash

# Example run: bash run_FT.sh [TASK] [EX_NAME_SUFFIX] [GPU_ID]

proj_dir=.

code_dir=${proj_dir}


# task and data
task_name=$1

# pretrain model
model_name_or_path=bert-base-uncased
# model_name_or_path=google/bert_uncased_L-6_H-768_A-12

# logging & saving
logging_steps=10
save_steps=0
eval_steps=5000

# train parameters
max_seq_length=384
batch_size=16
learning_rate=4e-5
epochs=10

# seed
seed=57

# output directory
ex_name_suffix=$2
ex_name=${task_name}_${ex_name_suffix}
output_dir=out/${task_name}/${ex_name}
mkdir -p $output_dir
pruning_type=None

CUDA_VISIBLE_DEVICES=$3 python3 $code_dir/run_qa_prune.py \
	   --output_dir ${output_dir} \
	   --logging_steps ${logging_steps} \
	   --model_name_or_path ${model_name_or_path} \
	   --ex_name ${ex_name} \
	   --do_train \
	   --do_eval \
	   --max_seq_length ${max_seq_length} \
	   --per_device_train_batch_size ${batch_size} \
	   --per_device_eval_batch_size 32 \
	   --learning_rate ${learning_rate} \
	   --num_train_epochs ${epochs} \
	   --overwrite_output_dir \
	   --save_steps ${save_steps} \
	   --eval_steps ${eval_steps} \
	   --evaluation_strategy steps \
       --version_2_with_negative \
       --doc_stride 128 \
	   --seed ${seed} 2>&1 | tee ${output_dir}/log.txt
