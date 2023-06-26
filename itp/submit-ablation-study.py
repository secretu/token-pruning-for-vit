import os
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map
import random
import time

def single_job(arg):
    task, sparsity, lr, reg_lr, warmup_epoch, epoch, bin_num, eval_step, topk, alpha, prune_location = arg
    # do it quietly
    command = f"python token_pruning.py submit --target sing_research --task {task} --sparsity {sparsity} --location {prune_location} --learning_rate {lr} --reg_learning_rate {reg_lr} --sparsity_reg_loss_alpha {alpha} --warmup_epochs {warmup_epoch} --bin_num {bin_num} --epochs {epoch} --eval_step {eval_step} --topk {topk} --tag 0129ablationNoPrunerGate"
    os.system(command)

# 0129ablationNoDiStill
# 0129ablationMSEDistill
# 0129ablationKLDistill
# 0129ablationNoPrunerGate
# 0129ablationNoAll

task =                      ["CoLA", "RTE", "QQP", "MRPC", "SST2", "MNLI", "QNLI", "STSB", "20news"]
sparsity =                  [0.43,   0.6,   0.65,  0.68,   0.4,    0.5,    0.58,   0.71,   0.81]
lr =                        [1e-5,   8e-5,  2e-5,  8e-5,   1e-5,   4e-5,   4e-5,   4e-5,   6e-5]
reg_lr =                    [0.1,    0.01,  0.01,  0.01,   0.01,   0.01,   0.02,   0.01,   0.01]
warmup_epochs =             [50,     50,    10,    80,     10,     10,     10,     50,     10]
epochs =                    [100,    100,   40,    100,    40,     40,     40,     100,    40]
bin_nums =                  [20,     100,   50,    50,     25,     50,     50,     30,     512]
eval_step =                 [50,     50,    3000,  50,     500,    2000,   500,    50,     300]
topk =                      [20] * len(task)
# sparsity_reg_loss_alpha =   [1e-2] * len(task)
sparsity_reg_loss_alpha =   [1.0] * len(task) # MSE and KL
prune_locations =           ["2,3,4,5,6,7,8,9,10,11"] * len(task)

for t, s, l, r, w, ep, b, ev, to, sp, pl in zip(task, sparsity, lr, reg_lr, warmup_epochs, epochs, bin_nums, eval_step, topk, sparsity_reg_loss_alpha, prune_locations):
    if t == "20news":
        args = [t, s, l, r, w, ep, b, ev, to, sp, pl]
        single_job(args)
