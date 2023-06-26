import os
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map
import random
import time

def single_job(arg):
    task, s, prune_location, l, r, alpha = arg
    # do it quietly
    command = f"python token_pruning.py submit --target sing_octo --task {task} --sparsity {s} --location {prune_location} --learning_rate {l} --reg_learning_rate {r} --sparsity_reg_loss_alpha {alpha} --warmup_epochs 10 --bin_num 35 --epochs 40 --eval_step 500 --tag 0129range"
    os.system(command)

task = "SST2"
sparsity = [0.57, 0.6]
prune_locations = ["2,3,4,5,6,7,8,9,10,11"]
lr = [6e-5, 4e-5, 2e-5]
reg_lr = [0.04, 0.02]
sparsity_reg_loss_alpha = [1e-2] # arp

args = []
for s in sparsity:
    for prune_location in prune_locations:
        for alpha in sparsity_reg_loss_alpha:
            for l in lr:
                for r in reg_lr:
                    args.append((task, s, prune_location, l, r, alpha))

print("total jobs:", len(args))
# single_job(args[0])
process_map(single_job, args, max_workers=1, chunksize=1)
print("all launched")
