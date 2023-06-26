import os
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map
import random
import time

def single_job(arg):
    task, s, prune_location, l, r, alpha = arg
    # do it quietly
    command = f"python token_pruning.py submit --target sing_octo --task {task} --sparsity {s} --location {prune_location} --learning_rate {l} --reg_learning_rate {r} --sparsity_reg_loss_alpha {alpha} --warmup_epochs 50 --epochs 100 --eval_step 50 --bin_num 20 --topk 10 --tag 0127scaledtopkARP2 > /dev/null 2>&1"
    os.system(command)

task = "CoLA"
sparsity = [0.43]
prune_locations = ["2,3,4,5,6,7,8,9,10,11"]
lr = [1e-5, 8e-6]
reg_lr = [0.1, 0.04, 0.02, 0.01]
# sparsity_reg_loss_alpha = [5e-3, 2e-3, 1e-3] # ndcg
# sparsity_reg_loss_alpha = [2e-4, 1e-4, 5e-5] # logistic
sparsity_reg_loss_alpha = [1e-3, 1e-4, 1e-5] # arp

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
