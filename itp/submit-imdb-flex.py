import os
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map
import random
import time
import numpy as np

def single_job(arg):
    task, s, prune_location, l, r, alpha, warmup_epoch = arg
    # do it quietly
    command = f"python token_pruning.py submit --target sing_research --task {task} --sparsity {s} --location {prune_location} --learning_rate {l} --reg_learning_rate {r} --sparsity_reg_loss_alpha {alpha} --topk 50 --warmup_epochs {warmup_epoch} --epochs 40 --eval_step 2000 --bin_num 256 --tag 0130"
    # print(command)
    # exit()
    os.system(command)

# ndcg 1e-2
# loss: 0.000604, lagrangian_loss: 0.000073, attention_score_distillation_loss: 0.024108
# loss: 0.771717, lagrangian_loss: 0.000782, attention_score_distillation_loss: 0.025462

task = "imdb"
sparsity = [0.55]
prune_locations = ["3,4,5,6,7,8,9,10,11"]
lr = [1e-5, 8e-6, 4e-6]
reg_lr = [0.01, 0.005]
sparsity_reg_loss_alpha = [1e-3, 1e-4] # arp
warmup_epochs = [20, 30]

args = []
for s in sparsity:
    for prune_location in prune_locations:
        for warmup_epoch in warmup_epochs:
            for alpha in sparsity_reg_loss_alpha:
                for l in lr:
                    for r in reg_lr:
                        args.append((task, s, prune_location, l, r, alpha, warmup_epoch))

print("total jobs:", len(args))
# single_job(args[0])
# single_job(args[1])
process_map(single_job, args, max_workers=1, chunksize=1)
print("all launched")
