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
  sku: G2
  priority: high
  command:
  - python sleep_and_print.py
"""

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('func', choices=['submit', 'debug'], help='submit job or local debug')
    parser.add_argument('--target', default='sing_octo', choices=list(target_dict.keys()), help='where to submit')
    # parser.add_argument('--sparsity', type=str, required=True)
    # parser.add_argument("--name", type=str, required=True)
    args = parser.parse_args()

    if args.func == 'submit':
        mode = 1
    else:
        mode = 0
    
    date = datetime.datetime.now().strftime('%m%d%s')
    job_name = f'cofipruning-sleep-{date}'
    jobs = job_template.format(
        job_name=job_name,
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
