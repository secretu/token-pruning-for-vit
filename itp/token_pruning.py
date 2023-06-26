import subprocess
import random
import string
import os
import argparse
import datetime

from modules.target import target_dict

template = \
"""

{target}

code:
  local_dir: $CONFIG_DIR/../../

storage:
  teamdrive:
    storage_account_name: hexnas
    container_name: teamdrive
    mount_dir: /mnt/data


jobs:
{jobs}
"""

job_template = \
"""- name: {job_name}
  sku: G1
  priority: high
  command:
  - bash run_joint_pruning_itp.sh {task} {sparsity} {location} {learning_rate} {reg_learning_rate} {sparsity_reg_loss_alpha} {warmup_epochs} {tag} {bin_num} {topk} {epochs} {eval_step}
"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('func', choices=['submit', 'debug'], help='submit job or local debug')
    parser.add_argument('--target', default='sing_octo', choices=list(target_dict.keys()), help='where to submit')
    parser.add_argument("--task", type=str, required=True)
    parser.add_argument("--location", type=str, default="2,3,4,5,6,7,8,9,10,11")
    parser.add_argument("--sparsity", type=float, default=0.5)
    parser.add_argument("--learning_rate", type=float, default=2.0e-5)
    parser.add_argument("--reg_learning_rate", type=float, default=0.01)
    parser.add_argument("--sparsity_reg_loss_alpha", type=float, default=1.0)
    parser.add_argument("--warmup_epochs", type=int, default=10)
    parser.add_argument("--bin_num", type=int, default=50)
    parser.add_argument("--tag", type=str, default=None)
    parser.add_argument("--topk", type=int, default=20)
    parser.add_argument("--epochs", type=int, required=True)
    parser.add_argument("--eval_step", type=int, required=True)
    args = parser.parse_args()

    if args.func == 'submit':
        mode = 1
    else:
        mode = 0

    date = datetime.datetime.now().strftime('%m%d%s')[-5:]
    task = args.task
    sparsity = args.sparsity
    location = args.location
    learning_rate = args.learning_rate
    reg_learning_rate = args.reg_learning_rate
    sparsity_reg_loss_alpha = args.sparsity_reg_loss_alpha
    warmup_epochs = args.warmup_epochs
    bin_num = args.bin_num
    topk = args.topk
    epochs = args.epochs
    eval_step = args.eval_step
    tag = args.tag if args.tag is not None else date
    job_name = f'token-pruning-{tag}-{task}-{sparsity}-lr{learning_rate}-reglr{reg_learning_rate}-regAlpha{sparsity_reg_loss_alpha}-warmup{warmup_epochs}-bin{bin_num}-topk{topk}-e{epochs}-{date}'
    jobs = job_template.format(
        job_name=job_name,
        task=task,
        sparsity=sparsity,
        location=location,
        learning_rate=learning_rate,
        reg_learning_rate=reg_learning_rate,
        sparsity_reg_loss_alpha=sparsity_reg_loss_alpha,
        warmup_epochs=warmup_epochs,
        bin_num=bin_num,
        tag=tag,
        topk=topk,
        epochs=epochs,
        eval_step=eval_step,
    )
    description = f'{job_name}'

    # ======================================================================================================
    # Don't need to modify following code
    result = template.format(
        job_name=job_name,
        jobs=jobs,
        target=target_dict[args.target]
    )   
    print(result)

    tmp_name = ''.join(random.choices(string.ascii_lowercase, k=6)) + job_name
    tmp_name = os.path.join("./.tmp", tmp_name)
    with open(tmp_name, "w") as fout:
        fout.write(result)

    if mode == 0:
        subprocess.run(["amlt", "run", "-t", "local", "--use-sudo", tmp_name, "--devices", "all"])
    else:
        # subprocess.run(f'amlt run -d {description} {tmp_name} {job_name}', shell=True)
        subprocess.run(["amlt", "run", "-d", description, tmp_name, job_name])

if __name__ == "__main__":
    main()
