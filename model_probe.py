from pathlib import Path
import argparse
import subprocess
from experiments.exp_def import (
    Experiment,
)

parser = argparse.ArgumentParser()
parser.add_argument('--model_ckpt', type=str, default='')
parser.add_argument('--task', type=str, default='')
args = parser.parse_args()

model_ckpt = Path(args.model_ckpt)
task_to_n_classes = {
    'NLI': 3,
    'POS': 17,
    'PAWSX': 2,
    'MARC': 5,
    'NER': 7
}
seeds = 1
devices = [6, 7]

processes = []
if args.task == '':
    tasks = list(Experiment)
else:
    tasks = [Experiment[args.task.upper()]]

for downstream_task in tasks:
    exp_name = f'{model_ckpt.parent.name}:{downstream_task.name}'
    dataset = f'{downstream_task.name}/cross'
    cmd = 'python train.py'
    cmd += f' --devices {len(processes)}'
    cmd += f' --model_probe --model_probe_n_classes {task_to_n_classes[downstream_task.name]}'
    cmd += f' --exp_name {exp_name}'
    cmd += f' --dataset_name {dataset}'
    cmd += ' --epochs 2'
    cmd += f' --model_ckpt {model_ckpt} --resume'

    process = subprocess.Popen(cmd, shell=True)
    processes.append(process)

    if len(processes) == 3:
        results = [p.wait() for p in processes]
        processes = []


