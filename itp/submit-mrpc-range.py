import os
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map
import random
import time

def single_job(arg):
    task, s, prune_location, l, r, alpha, bin_num, topk = arg
    # do it quietly
    command = f"python token_pruning.py submit --target sing_octo --task {task} --sparsity {s} --location {prune_location} --learning_rate {l} --reg_learning_rate {r} --sparsity_reg_loss_alpha {alpha} --warmup_epochs 80 --bin_num {bin_num} --topk {topk} --epochs 100 --eval_step 50 --tag 0201rank"
    # print(command)
    # exit()
    os.system(command)

task = "MRPC"
sparsity = [0.5, 0.7]
# sparsity = [0.75]
prune_locations = ["2,3,4,5,6,7,8,9,10,11"]
lr = [8e-5, 6e-5]
reg_lr = [0.04, 0.02, 0.01]
# sparsity_reg_loss_alpha = [2e-3, 1e-3, 1e-4] # ndcg
# sparsity_reg_loss_alpha = [1e-4, 5e-5, 1e-5] # logistic
# sparsity_reg_loss_alpha = [5e-2, 1e-2]
# sparsity_reg_loss_alpha = [1e-1, 1e-2, 1e-3] # ndcg@60
sparsity_reg_loss_alpha = [1e-2, 1e-3, 1e-4] # scaled topk arp
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
