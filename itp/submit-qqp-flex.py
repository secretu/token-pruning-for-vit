import os
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map
import random
import time

def single_job(arg):
    task, s, prune_location, l, r, alpha, warmup_epoch, topk = arg
    # do it quietly
    command = f"python token_pruning.py submit --target sing_research --task {task} --sparsity {s} --location {prune_location} --learning_rate {l} --reg_learning_rate {r} --sparsity_reg_loss_alpha {alpha} --warmup_epochs {warmup_epoch} --bin_num 50 --topk {topk} --epochs 40 --eval_step 3000 --tag 0129"
    os.system(command)

task = "QQP"
sparsity = [0.45]
prune_locations = ["2,3,4,5,6,7,8,9,10,11", "3,4,5,6,7,8,9,10,11"]
lr = [2e-5]
reg_lr = [0.01]
sparsity_reg_loss_alpha = [1e-2, 1e-4] # scaled arp 
warmup_epochs = [10]
topks = [20]

args = []
for s in sparsity:
    for prune_location in prune_locations:
        for topk in topks:
            for warmup_epoch in warmup_epochs:
                for alpha in sparsity_reg_loss_alpha:
                    for l in lr:
                        for r in reg_lr:
                            args.append((task, s, prune_location, l, r, alpha, warmup_epoch, topk))

print("total jobs:", len(args))
process_map(single_job, args, max_workers=1, chunksize=1)
print("all launched")
