import os
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map
import random
import time

def single_job(arg):
    task, s, prune_location, l, r, alpha, bin_num = arg
    # do it quietly
    command = f"python token_pruning.py submit --target sing_research --task {task} --sparsity {s} --location {prune_location} --learning_rate {l} --reg_learning_rate {r} --sparsity_reg_loss_alpha {alpha} --warmup_epochs 10 --bin_num {bin_num} --epochs 40 --eval_step 500 --tag 0129ablationBin > /dev/null 2>&1"
    os.system(command)

task = "SST2"
sparsity = [0.4]
prune_locations = ["2,3,4,5,6,7,8,9,10,11"]
lr = [2e-5, 1e-5]
reg_lr = [0.02, 0.01]
sparsity_reg_loss_alpha = [1e-2] # arp
bin_nums = [20, 35, 45, 64]

args = []
for s in sparsity:
    for prune_location in prune_locations:
        for bin_num in bin_nums:
            for alpha in sparsity_reg_loss_alpha:
                for l in lr:
                    for r in reg_lr:
                        args.append((task, s, prune_location, l, r, alpha, bin_num))

print("total jobs:", len(args))
# single_job(args[0])
process_map(single_job, args, max_workers=1, chunksize=1)
print("all launched")
