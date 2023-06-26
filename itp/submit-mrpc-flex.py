import os
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map
import random
import time

def single_job(arg):
    task, s, prune_location, l, r, alpha, bin_num, topk = arg
    # do it quietly
    command = f"python token_pruning.py submit --target sing_research --task {task} --sparsity {s} --location {prune_location} --learning_rate {l} --reg_learning_rate {r} --sparsity_reg_loss_alpha {alpha} --warmup_epochs 150 --bin_num {bin_num} --topk {topk} --epochs 200 --eval_step 50 --tag 0125scaledtopkARP2"
    # print(command)
    # exit()
    os.system(command)

task = "MRPC"
# sparsity = [0.68]
# prune_locations = ["1,2,3,4,5,6,7,8,9,10,11", "2,3,4,5,6,7,8,9,10,11"]
# lr = [8e-5, 6e-5, 4e-5, 2e-5]
# reg_lr = [0.1, 0.05, 0.02, 0.01]
# # sparsity_reg_loss_alpha = [0.01, 0.005, 0.002, 0.001] # ndcg
# # sparsity_reg_loss_alpha = [0.0001, 0.00001, 0.000001] # hinge
# sparsity_reg_loss_alpha = [1.0]
# bin_nums = [50]

sparsity = [0.68]
prune_locations = ["2,3,4,5,6,7,8,9,10,11"]
lr = [8e-5, 6e-5]
reg_lr = [0.04, 0.02, 0.01]
# sparsity_reg_loss_alpha = [2e-3, 1e-3, 1e-4] # ndcg
# sparsity_reg_loss_alpha = [1e-4, 5e-5, 1e-5] # logistic
# sparsity_reg_loss_alpha = [5e-2, 1e-2]
# sparsity_reg_loss_alpha = [1e-1, 1e-2, 1e-3] # ndcg@60
sparsity_reg_loss_alpha = [1e-2, 1e-3, 1e-4, 1e-5] # scaled topk arp
bin_nums = [50]
topks = [20]

args = []
for s in sparsity:
    for prune_location in prune_locations:
        for topk in topks:
            for bin_num in bin_nums:
                for alpha in sparsity_reg_loss_alpha:
                    for l in lr:
                        for r in reg_lr:
                            args.append((task, s, prune_location, l, r, alpha, bin_num, topk))

print("total jobs:", len(args))
# single_job(args[0])
# single_job(args[1])
process_map(single_job, args, max_workers=1, chunksize=1)
print("all launched")
