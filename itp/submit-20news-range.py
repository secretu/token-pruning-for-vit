import os
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map
import random
import time
import numpy as np

def single_job(arg):
    task, s, prune_location, l, r, alpha = arg
    # do it quietly
    command = f"python token_pruning.py submit --target sing_octo --task {task} --sparsity {s} --location {prune_location} --learning_rate {l} --reg_learning_rate {r} --sparsity_reg_loss_alpha {alpha} --warmup_epochs 10 --bin_num 512 --topk 60 --epochs 50 --eval_step 1000 --tag 0201range > /dev/null 2>&1"
    # print(command)
    # exit()
    os.system(command)

task = "20news"
# sparsity = [0.77] # x4.22
# sparsity = [0.5, 0.6, 0.7, 0.85, 0.9]
sparsity = [0.82, 0.83]
prune_locations = ["2,3,4,5,6,7,8,9,10,11"]
lr = [8e-5, 6e-5, 4e-5, 2e-5]
reg_lr = [0.04, 0.02, 0.01]
# sparsity_reg_loss_alpha = [5e-3, 2e-3, 1e-3] # ndcg
# sparsity_reg_loss_alpha = [2e-4, 1e-4, 5e-5] # logistic
sparsity_reg_loss_alpha = [1e-3, 1e-4] # arp
# sparsity_reg_loss_alpha = [1e-2, 1e-3, 1e-4, 1e-5] # ndcg@60

args = []
for s in sparsity:
    for prune_location in prune_locations:
        for alpha in sparsity_reg_loss_alpha:
            for l in lr:
                for r in reg_lr:
                    args.append((task, s, prune_location, l, r, alpha))

print("total jobs:", len(args))
# single_job(args[0])
# single_job(args[1])
process_map(single_job, args, max_workers=1, chunksize=1)
print("all launched")
