import joblib
from experiments.exp_def import Experiment, LingualSetting, TaskDefs
from pathlib import Path
from get_model_probe_results import get_model_probe_scores, get_model_probe_final_score, combine_all_model_probe_scores
from train import main
from data_utils.log_wrapper import create_logger
from joblib import Parallel, delayed
import os

specs = {
    "GOEMOTIONS": {
        'batch_size': 64,
        'learning_rate': 5e-3,
        'metric_of_choice': "F1MAC", 
        'log_per_updates': 100,  
        'is_model_probe_sequence': False, 
    }, 
    "NER": {
        'batch_size': 64,
        'learning_rate': 5e-2,
        'metric_of_choice': 'SeqEvalList', 
        'log_per_updates': 100,   
        'is_model_probe_sequence': True, 
    },
    "POS": {
        'batch_size': 64,
        'learning_rate': 5e-3,
        'metric_of_choice': 'SeqEvalList', 
        'log_per_updates': 100,   
        'is_model_probe_sequence': True, 
    },
    "SST5": {
        'batch_size': 32,
        'learning_rate': 5e-3,
        'metric_of_choice': 'F1MAC', 
        'log_per_updates': 100,   
        'is_model_probe_sequence': False, 
    },
    "PAWS": {
        'batch_size': 64,
        'learning_rate': 5e-4,
        'metric_of_choice': 'F1',
        'log_per_updates': 100,   
        'is_model_probe_sequence': False, 
    },
    "COLA": {
        'batch_size': 32,
        'learning_rate': 5e-3,
        'metric_of_choice': 'MCC', 
        'log_per_updates': 100,   
        'is_model_probe_sequence': False, 
    },
    "MRPC": {
        'batch_size': 32,
        'learning_rate': 5e-3,
        'metric_of_choice': 'F1', 
        'log_per_updates': 100,   
        'is_model_probe_sequence': False, 
    },
    "SCITAIL": {
        'batch_size': 64,
        'learning_rate': 5e-3,
        'metric_of_choice': 'F1', 
        'log_per_updates': 100,   
        'is_model_probe_sequence': False, 
    },
    "WNLI": {
        'batch_size': 16,
        'learning_rate': 5e-5,
        'metric_of_choice': 'F1', 
        'log_per_updates': 10,   
        'is_model_probe_sequence': False, 
    },
    "SST2": {
        'batch_size': 64,
        'learning_rate': 5e-3,
        'metric_of_choice': 'F1', 
        'log_per_updates': 100,   
        'is_model_probe_sequence': False, 
    },
    "EMOTION": {
        'batch_size': 32,
        'learning_rate': 5e-3,
        'metric_of_choice': 'F1MAC', 
        'log_per_updates': 100,   
        'is_model_probe_sequence': False, 
    },
}

SEED=2022

print(f'\nSTARTED FINETUNING...\n')

log_path = f'logs/seed_{SEED}/mt-dnn.log'
Path(log_path).parent.mkdir(parents=True, exist_ok=True)
logger = create_logger(__name__, to_disk=True, log_file=log_path)
os.environ["WANDB_START_METHOD"] = "thread"

jobs = [
    delayed(main)(f'--bert_pretrained --seed {SEED} --devices 0 --exp_name COARSE_base --dataset_name COARSE/en --epochs 5 --batch_size 16 --learning_rate 5e-4 --grad_accumulation_step 4 --bert_model_type bert-base-cased --fp16 --metric_of_choice MCC --resume --log_per_updates 10 --huggingface_ckpt --model_ckpt /home/aadil/mt-dnn/mt_dnn_models/bert_model_base_cased.pt', logger),
    delayed(main)(f'--seed {SEED} --devices 1 --exp_name COARSE_cross --dataset_name COARSE/en --wandb --epochs 5 --batch_size 16 --learning_rate 5e-4 --grad_accumulation_step 4 --bert_model_type bert-base-cased --fp16 --metric_of_choice MCC --resume --log_per_updates 10 --huggingface_ckpt --model_ckpt /home/aadil/mt-dnn/mt_dnn_models/bert_model_base_cased.pt', logger),
    delayed(main)(f'--seed {SEED} --devices 2 --exp_name FINER_cross --dataset_name FINER/en --wandb --epochs 5 --batch_size 16 --learning_rate 5e-4 --grad_accumulation_step 4 --bert_model_type bert-base-cased --fp16 --metric_of_choice MCC --resume --log_per_updates 10 --huggingface_ckpt --model_ckpt /home/aadil/mt-dnn/mt_dnn_models/bert_model_base_cased.pt', logger)
]
results = Parallel(n_jobs=joblib.cpu_count())(jobs)

print(f'\n...Finished FINETUNING\n')

#for task in [Experiment.COLA, Experiment.MRPC, Experiment.SCITAIL, Experiment.WNLI, Experiment.PAWS, Experiment.SST2, Experiment.EMOTION]:
for task in [Experiment.GOEMOTIONS, Experiment.NER, Experiment.POS, Experiment.SST5, Experiment.COLA, Experiment.MRPC, Experiment.SCITAIL, Experiment.WNLI, Experiment.PAWS, Experiment.SST2, Experiment.EMOTION]:
    print(f'\nSTARTED DOWNSTREAM: {task.name}...\n')
    task_def_path = Path('experiments').joinpath(task.name, 'task_def.yaml')
    task_def = TaskDefs(task_def_path).get_task_def(task.name.lower())
    model_probe_sequence = '--model_probe_sequence' if specs[task.name]['is_model_probe_sequence'] else ''
    jobs = [
        delayed(main)(f'--seed {SEED} --devices 3 --model_probe --model_probe_n_classes {task_def["n_class"]} {model_probe_sequence} --exp_name COARSE_base_{task.name} --dataset_name {task.name}/en --wandb --epochs 2 --batch_size {specs[task.name]["batch_size"]} --learning_rate {specs[task.name]["learning_rate"]} --bert_model_type bert-base-cased --model_ckpt checkpoint/COARSE_base/model_0_0.pt --metric_of_choice {specs[task.name]["metric_of_choice"]} --log_per_updates {specs[task.name]["log_per_updates"]} --fp16', logger),
        delayed(main)(f'--seed {SEED} --devices 2 --model_probe --model_probe_n_classes {task_def["n_class"]} {model_probe_sequence} --exp_name COARSE_{task.name} --dataset_name {task.name}/en --wandb --epochs 2 --batch_size {specs[task.name]["batch_size"]} --learning_rate {specs[task.name]["learning_rate"]} --bert_model_type bert-base-cased --model_ckpt checkpoint/COARSE_cross/model_4_125.pt --metric_of_choice {specs[task.name]["metric_of_choice"]} --log_per_updates {specs[task.name]["log_per_updates"]} --fp16', logger),
        delayed(main)(f'--seed {SEED} --devices 1 --model_probe --model_probe_n_classes {task_def["n_class"]} {model_probe_sequence} --exp_name FINER_{task.name} --dataset_name {task.name}/en --wandb --epochs 2 --batch_size {specs[task.name]["batch_size"]} --learning_rate {specs[task.name]["learning_rate"]} --bert_model_type bert-base-cased --model_ckpt checkpoint/FINER_cross/model_4_125.pt --metric_of_choice {specs[task.name]["metric_of_choice"]} --log_per_updates {specs[task.name]["log_per_updates"]} --fp16', logger)
    ]
    results = Parallel(n_jobs=joblib.cpu_count())(jobs)
    for (finetuned_task, finetuned_setting) in [(Experiment.COARSE, LingualSetting.BASE), (Experiment.COARSE, LingualSetting.CROSS), (Experiment.FINER, LingualSetting.CROSS)]:
        helper = '_base' if finetuned_setting is LingualSetting.BASE else ''
        get_model_probe_scores(
            finetuned_task=finetuned_task,
            finetuned_setting=finetuned_setting,
            probe_setting=LingualSetting.CROSS,
            probe_task=task,
            model_ckpt=list(Path(f'checkpoint/{finetuned_task.name}{helper}_{task.name}').rglob('*.pt'))[0],
            out_file_name='results.csv',
            metric_of_choice=specs[task.name]['metric_of_choice'],
            device_id=0,
            lang='en',
            seed=SEED,
            batch_size=64,
            max_seq_len=512,
        )
    print(f'\n...FINISHED DOWNSTREAM: {task.name}\n')

print(f'\nCOMBINING ALL MODEL PROBE SCORES...\n')
get_model_probe_final_score(Experiment.COARSE, LingualSetting.CROSS)
get_model_probe_final_score(Experiment.FINER, LingualSetting.CROSS)
combine_all_model_probe_scores(mean=True, std=True)
print(f'\n...FINISHED COMBINING ALL MODEL PROBE SCORES\n')

print('DONE!')
