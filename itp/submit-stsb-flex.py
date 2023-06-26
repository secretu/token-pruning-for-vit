import os
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map
import random
import time

def single_job(arg):
    task, s, prune_location, l, r, alpha, bin_num, epoch = arg
    # do it quietly
    command = f"python token_pruning.py submit --target sing_octo --task {task} --sparsity {s} --location {prune_location} --learning_rate {l} --reg_learning_rate {r} --sparsity_reg_loss_alpha {alpha} --warmup_epochs 50 --bin_num {bin_num} --topk 20 --epochs {epoch} --eval_step 100 --tag 020101 > /dev/null 2>&1"
    # print(command)
    # exit()
    os.system(command)

task = "STSB"
sparsity = [0.55, 0.6]
prune_locations = ["3,4,5,6,7,8,9,10,11", "2,3,4,5,6,7,8,9,10,11"]
lr = [6e-5, 4e-5, 2e-5, 1e-5]
reg_lr = [0.1, 0.04, 0.01]
sparsity_reg_loss_alpha = [1e-2, 1e-3, 1e-4] # scaled arp
epochs = [100]
bin_nums = [30]

args = []
for s in sparsity:
    for bin_num in bin_nums:
        for epoch in epochs:
            for prune_location in prune_locations:
                for alpha in sparsity_reg_loss_alpha:
                    for l in lr:
                        for r in reg_lr:
                            args.append((task, s, prune_location, l, r, alpha, bin_num, epoch))

print("total jobs:", len(args))
single_job(args[0])
process_map(single_job, args, max_workers=2, chunksize=1)
print("all launched")
