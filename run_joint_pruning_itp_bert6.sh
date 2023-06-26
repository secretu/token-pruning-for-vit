classification_tasks=(20news yelp imdb)
qa_tasks=(squad_v2 squad_v2)
TASK=${1}
SPARSITY=${2}
PRUNE_LOCATION=${3}
LEARNING_RATE=${4}
REG_LEARNING_RATE=${5}
# mask loss alpha
DISTILL_CE_LOSS_ALPHA=${6}
WARMUP_EPOCHS=${7}
BIN_NUM=${9}
TOPK=${10}
EPOCHS=${11}
EVAL_STEP=${12}
SUFFIX=s${SPARSITY}_lr${LEARNING_RATE}_reglr${REG_LEARNING_RATE}_alpha${DISTILL_CE_LOSS_ALPHA}_warmup${WARMUP_EPOCHS}_bin${BIN_NUM}
EX_CATE=${8}
# joint pruning
# PRUNING_TYPE=structured_heads+structured_mlp+head_layer+token
# model pruning
# PRUNING_TYPE=structured_heads+structured_mlp+head_layer
# token pruning
PRUNING_TYPE=token+pruner
DISTILL_LAYER_LOSS_ALPHA=0.9
LAYER_DISTILL_VERSION=4
SPARSITY_EPSILON=0.01
DISTILLATION_PATH=/mnt/data/device-aware-bert/token_pruning/teachers/BERT6/${TASK}

script_file=run_TokenPruneITP.sh
if [[ " ${classification_tasks[*]} " =~ ${TASK} ]]; then
    script_file=run_TokenPruneITP_classification.sh
fi

if [[ " ${qa_tasks[*]} " =~ ${TASK} ]]; then
    script_file=run_TokenPruneITP_qa.sh
fi

bash scripts/${script_file} $TASK $SUFFIX $EX_CATE $PRUNING_TYPE $SPARSITY $DISTILLATION_PATH $DISTILL_LAYER_LOSS_ALPHA $DISTILL_CE_LOSS_ALPHA $LAYER_DISTILL_VERSION $SPARSITY_EPSILON $PRUNE_LOCATION $LEARNING_RATE $REG_LEARNING_RATE $WARMUP_EPOCHS $BIN_NUM $TOPK $EPOCHS $EVAL_STEP
