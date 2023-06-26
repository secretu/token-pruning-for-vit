import os
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map
import random
import time

def single_job(arg):
    task, s, prune_location, l, r, alpha, warmup_epoch = arg
    # do it quietly
    command = f"python token_pruning.py submit --target sing_octo --task {task} --sparsity {s} --location {prune_location} --learning_rate {l} --reg_learning_rate {r} --sparsity_reg_loss_alpha {alpha} --warmup_epochs {warmup_epoch} --bin_num 150 --epochs 10 --eval_step 1000 --topk 60 --tag 0202goodluck"
    os.system(command)

# 20epochs 219600
task = "squad_v2"
sparsity = [0.48, 0.5]
prune_locations = ["3,4,5,6,7,8,9,10,11"]
lr = [6e-5, 4e-5]
reg_lr = [0.04, 0.02, 0.01]
sparsity_reg_loss_alpha = [1e-2] # scaled arp
warmup_epochs = [2, 5]

args = []
for s in sparsity:
    for prune_location in prune_locations:
        for warmup_epoch in warmup_epochs:
            for alpha in sparsity_reg_loss_alpha:
                for l in lr:
                    for r in reg_lr:
                        args.append((task, s, prune_location, l, r, alpha, warmup_epoch))

print("total jobs:", len(args))
process_map(single_job, args, max_workers=1, chunksize=1)
print("all launched")
