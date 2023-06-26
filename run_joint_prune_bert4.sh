TASK=STSB
SPARSITY=0.6
PRUNE_LOCATION=1,2,3
SUFFIX=sparsity${SPARSITY}
EX_CATE=debugdebug
# joint pruning
# PRUNING_TYPE=structured_heads+structured_mlp+head_layer+token
# model pruning
# PRUNING_TYPE=structured_heads+structured_mlp+head_layer
# token pruning
PRUNING_TYPE=token+pruner
DISTILL_LAYER_LOSS_ALPHA=0.9
DISTILL_CE_LOSS_ALPHA=100.0
LAYER_DISTILL_VERSION=4
SPARSITY_EPSILON=0.01
DISTILLATION_PATH=out/${TASK}/${TASK}_BERT4/best

bash scripts/run_JointPrune.sh $TASK $SUFFIX $EX_CATE $PRUNING_TYPE $SPARSITY $DISTILLATION_PATH $DISTILL_LAYER_LOSS_ALPHA $DISTILL_CE_LOSS_ALPHA $LAYER_DISTILL_VERSION $SPARSITY_EPSILON $PRUNE_LOCATION
