#!/bin/bash
#SBATCH --job-name=sample
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH -A pnlp
#SBATCH -t 11:00:00


glue_low=(MRPC RTE STSB CoLA)
glue_high=(MNLI QQP QNLI SST2)

proj_dir=.

code_dir=${proj_dir}

# task and data
task_name=$1
data_dir=$proj_dir/data/glue_data/${task_name}

# pretrain model
model_name_or_path=out/${task_name}/${task_name}_full/best

# logging & saving
logging_steps=10
save_steps=0


# train parameters
max_seq_length=128
batch_size=32
learning_rate=2.0e-5
reg_learning_rate=0.01
epochs=20 

# seed
seed=57

# output dir
ex_name_suffix=$2
ex_name=${task_name}_${ex_name_suffix}
ex_cate=$3
output_dir=${proj_dir}/out/${task_name}/${ex_cate}/${ex_name}

# pruning and distillation
pruning_type=$4
target_sparsity=$5
distillation_path=$6
distill_layer_loss_alpha=$7
distill_ce_loss_alpha=$8
distill_temp=2
# 2: fix hidden layers, 3: min distance matching without restriction, 4: min distance matching with restriction
layer_distill_version=${9} 

# token pruning arguments
prune_location=1,2,3,4,5,6,7,8,9,10,11

scheduler_type=linear


if [[ " ${glue_low[*]} " =~ ${task_name} ]]; then
    eval_steps=50
    epochs=100
    start_saving_best_epochs=50
    prepruning_finetune_epochs=4
    lagrangian_warmup_epochs=50
fi

if [[ " ${glue_high[*]} " =~ ${task_name} ]]; then
    eval_steps=500
    prepruning_finetune_epochs=1
    lagrangian_warmup_epochs=2
fi

pretrained_pruned_model=None

# FT after pruning
if [[ $pruning_type == None ]]; then
  pretrained_pruned_model=${13}
  learning_rate=${12}
  scheduler_type=linear
  output_dir=$pretrained_pruned_model/FT-lr${learning_rate}
  epochs=20
  batch_size=64
fi

mkdir -p $output_dir


#  --use_mac_l0
export TOKENIZERS_PARALLELISM=TRUE
python3 $code_dir/run_glue_prune.py \
	   --output_dir ${output_dir} \
	   --logging_steps ${logging_steps} \
	   --task_name ${task_name} \
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
	   --seed ${seed} \
	   --pruning_type ${pruning_type} \
     --pretrained_pruned_model ${pretrained_pruned_model} \
     --freeze_embeddings \
     --scheduler_type $scheduler_type \
     --dataloader_num_workers 8 \
     --log_level error \
     --prune_location $prune_location \
     --lagrangian_warmup_epochs $lagrangian_warmup_epochs 2>&1 | tee ${output_dir}/log.txt
