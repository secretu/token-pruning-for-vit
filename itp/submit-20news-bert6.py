import os
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map
import random
import time

def single_job(arg):
    task, s, prune_location, l, r, alpha, bin_num, topk = arg
    # do it quietly
    command = f"python token_pruning_bert6.py submit --target sing_octo --task {task} --sparsity {s} --location {prune_location} --learning_rate {l} --reg_learning_rate {r} --sparsity_reg_loss_alpha {alpha} --warmup_epochs 10 --bin_num {bin_num} --topk {topk} --epochs 20 --eval_step 1000 --tag 0131BERT6 > /dev/null 2>&1"
    # print(command)
    # exit()
    os.system(command)


task = "20news"
sparsity = [0.73, 0.74]
prune_locations = ["2,3,4,5"]
lr = [8e-5, 6e-5, 4e-5]
reg_lr = [0.04, 0.01]
sparsity_reg_loss_alpha = [1e-3, 1e-4] # arp
bin_nums = [512]
topks = [60]

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
