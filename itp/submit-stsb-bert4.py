import os
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map
import random
import time

def single_job(arg):
    task, s, prune_location, l, r, alpha = arg
    # do it quietly
    command = f"python token_pruning_bert4.py submit --target sing_octo --task {task} --sparsity {s} --location {prune_location} --learning_rate {l} --reg_learning_rate {r} --sparsity_reg_loss_alpha {alpha} --warmup_epochs 50 --tag 010202BERT4 > /dev/null 2>&1"
    os.system(command)

task = "STSB"
# sparsity = [0.3, 0.4, 0.5, 0.6]
sparsity = [0.4, 0.5, 0.6]
prune_locations = ["1,2,3"]
lr = [8e-5, 6e-5, 4e-5, 2e-5, 1e-5]
reg_lr = [0.05, 0.02, 0.01]
sparsity_reg_loss_alpha = [100.0, 200.0]

args = []
for s in sparsity:
    for prune_location in prune_locations:
        for alpha in sparsity_reg_loss_alpha:
            for l in lr:
                for r in reg_lr:
                    args.append((task, s, prune_location, l, r, alpha))

print("total jobs:", len(args))
single_job(args[0])
process_map(single_job, args, max_workers=2, chunksize=1)
print("all launched")
