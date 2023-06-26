import os
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map
import random
import time

def single_job(arg):
    task, s, prune_location, l, r, alpha, warmup_epoch = arg
    # do it quietly
    command = f"python token_pruning.py submit --target itp_rr1 --task {task} --sparsity {s} --location {prune_location} --learning_rate {l} --reg_learning_rate {r} --sparsity_reg_loss_alpha {alpha} --warmup_epochs {warmup_epoch} --bin_num 50 --epochs 40 --eval_step 2000 --topk 20 --tag 0127scaledtopkARP2"
    os.system(command)

# 20epoch 245440
task = "MNLI"
sparsity = [0.5]
prune_locations = ["2,3,4,5,6,7,8,9,10,11"]
lr = [4e-5]
reg_lr = [0.01]
sparsity_reg_loss_alpha = [1e-2, 1e-3, 1e-4, 1e-5] # scaled arp 
# sparsity_reg_loss_alpha = [1e-1, 1e-2, 1e-3] # scaled ndcg@60 (1e-1)
warmup_epochs = [10]

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
