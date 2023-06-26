classification_tasks=(20news yelp imdb)
qa_tasks=(squad_v2 squad_v2)
TASK=MRPC
SPARSITY=0.68
PRUNE_LOCATION=2,3,4,5,6,7,8,9,10,11 # keep this unchanged
SUFFIX=sparsity${SPARSITY}
EX_CATE=debug111
# joint pruning
# PRUNING_TYPE=structured_heads+structured_mlp+head_layer+token
# model pruning
# PRUNING_TYPE=structured_heads+structured_mlp+head_layer
# token pruning
PRUNING_TYPE=token+pruner
DISTILL_LAYER_LOSS_ALPHA=0.9 # this is not used
DISTILL_CE_LOSS_ALPHA=1.0 # change this for sparsity_reg_loss_alpha
LAYER_DISTILL_VERSION=4
SPARSITY_EPSILON=0.01
DISTILLATION_PATH=out/${TASK}/${TASK}_full/best
BIN_NUM=200
TOPK=50

script_file=run_JointPrune.sh
if [[ " ${classification_tasks[*]} " =~ ${TASK} ]]; then
    script_file=run_JointPrune_classification.sh
fi

if [[ " ${qa_tasks[*]} " =~ ${TASK} ]]; then
    script_file=run_JointPrune_qa.sh
fi

bash scripts/${script_file} $TASK $SUFFIX $EX_CATE $PRUNING_TYPE $SPARSITY $DISTILLATION_PATH $DISTILL_LAYER_LOSS_ALPHA $DISTILL_CE_LOSS_ALPHA $LAYER_DISTILL_VERSION $SPARSITY_EPSILON $PRUNE_LOCATION $BIN_NUM $TOPK
