#!/bin/bash
proj_dir=.

code_dir=${proj_dir}

# task and data
task_name=$1
data_dir=$proj_dir/data/glue_data/${task_name}

# pretrain model
model_name_or_path=$6

# logging & saving
logging_steps=10
save_steps=0


# train parameters
max_seq_length=384
batch_size=12
learning_rate=4.0e-5
reg_learning_rate=0.01
eval_steps=50
epochs=20
prepruning_finetune_epochs=1
lagrangian_warmup_epochs=10

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
prune_location=${11}
bin_num=${12}
topk=${13}

scheduler_type=linear

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

export TOKENIZERS_PARALLELISM=false
python3 $code_dir/run_qa_prune.py \
	   --output_dir ${output_dir} \
	   --logging_steps ${logging_steps} \
     --version_2_with_negative \
     --doc_stride 128 \
	   --model_name_or_path ${model_name_or_path} \
	   --ex_name ${ex_name} \
	   --do_train \
	   --do_eval \
	   --max_seq_length ${max_seq_length} \
	   --per_device_train_batch_size ${batch_size} \
	   --per_device_eval_batch_size 32 \
	   --learning_rate ${learning_rate} \
	   --reg_learning_rate ${reg_learning_rate} \
	   --num_train_epochs ${epochs} \
	   --overwrite_output_dir \
	   --save_steps ${save_steps} \
	   --eval_steps ${eval_steps} \
	   --evaluation_strategy steps \
	   --seed ${seed} \
	   --pruning_type ${pruning_type} \
     --pretrained_pruned_model ${pretrained_pruned_model} \
     --target_sparsity $target_sparsity \
     --freeze_embeddings \
     --do_distill \
     --distillation_path $distillation_path \
     --distill_ce_loss_alpha $distill_ce_loss_alpha \
     --distill_loss_alpha $distill_layer_loss_alpha \
     --distill_temp $distill_temp \
     --scheduler_type $scheduler_type \
     --layer_distill_version $layer_distill_version \
     --prepruning_finetune_epochs $prepruning_finetune_epochs \
     --use_mac_l0 \
     --dataloader_num_workers 8 \
     --log_level error \
     --droprate_init 0.01 \
     --temperature 0.8 \
     --prune_location $prune_location \
     --bin_num $bin_num \
     --topk $topk \
     --lagrangian_warmup_epochs $lagrangian_warmup_epochs 2>&1 | tee ${output_dir}/log.txt
