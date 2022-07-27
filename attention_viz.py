from attention_flow import get_attention
from experiments.exp_def import LingualSetting, Experiment, TaskDef, TaskDefs
from mt_dnn.model import MTDNNModel
from mt_dnn.batcher import SingleTaskDataset, Collater
from data_utils.task_def import EncoderModelType
from run_bertology import compute_heads_importance

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from pathlib import Path
import einops
from apex import amp
from transformers import BertTokenizer
from transformers.trainer_utils import set_seed

from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import string
import numpy as np
import argparse
import os

def plot_config(parser):
    parser.add_argument('--font_scale', type=float, default=0.7)
    parser.add_argument('--figure_width', type=int, default=15)
    parser.add_argument('--figure_height', type=int, default=7)
    parser.add_argument('--cbar', action='store_true')
    return parser

def model_config(parser):
    parser.add_argument('--finetuned_task', type=str, default='')
    parser.add_argument('--finetuned_setting', type=str, default='base')
    parser.add_argument('--output_dir', type=str, default='attention_outputs')
    parser.add_argument('--seed', type=int, default=2022)

    parser.add_argument('--compute_heads_importance', action='store_true')
    parser.add_argument(
        "--dont_normalize_importance_by_layer", action="store_true", help="Don't normalize importance score by layers"
    )
    parser.add_argument(
        "--dont_normalize_global_importance",
        action="store_true",
        help="Don't normalize all importance scores between 0 and 1",
    )
    parser.add_argument('--n_trials', type=int, default=5)

    parser.add_argument('--idx', nargs='*')
    parser.add_argument('--layer', type=int, default=1)
    parser.add_argument('--head', type=int, default=1)
    
    parser.add_argument('--devices', nargs='+')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--bert_model_type', type=str, default='bert-base-cased')
    return parser

def load_data(data_file, task_def, language, opt, is_train=True):
    """
    Create the test dataloader to use.

    Args:
    data_file: The JSON test data file
    task_def: Path to the task_def.
    language: Specific language to use.
    """
    test_data_set = SingleTaskDataset(
        data_file,
        is_train=is_train,
        task_def=task_def,
    )

    collater = Collater(is_train=is_train, encoder_type=EncoderModelType.BERT, soft_label=False)
    test_data = DataLoader(
        test_data_set,
        batch_size=opt['batch_size'],
        collate_fn=collater.collate_fn,
        pin_memory=True)

    return test_data

def create_model_and_dataloader(task: Experiment, setting: LingualSetting, opt: dict, device_id: int=0, is_train: bool=True):
    """
    Create the MT-DNN model, finetuend on finetuned_task in the {base, cross}-lingual setting.
    """
    checkpoint_dir = Path('checkpoint').joinpath(f'{task.name}_{setting.name.lower()}')
    checkpoint_file = list(checkpoint_dir.rglob('*.pt'))[0]
    state_dict = torch.load(checkpoint_file)
    del state_dict['optimizer']

    language = 'en'
    task_def_path = Path(f'experiments').joinpath(
        task.name,
        language,
        'task_def.yaml')
    task_def = TaskDefs(task_def_path).get_task_def(task.name.lower())
    task_def_list = [task_def]
    config = state_dict['config']
    config['task_def_list'] = task_def_list
    config['bin_on'] = False
    config['batch_size'] = opt['batch_size']

    model = MTDNNModel(config, state_dict=state_dict, devices=[device_id])
    model.network.eval()

    data_file = Path(f'experiments/{task.name}/').joinpath(language, opt['bert_model_type'], f'{task.name.lower()}_test.json')
    test_data = load_data(data_file, task_def, language, opt, is_train)

    return model, config, test_data

def get_attention_and_input_ids(model: MTDNNModel, data_loader: DataLoader, device_id: int=0):
    with torch.no_grad():
        attentions_lst = []
        input_ids_lst = []
        for batch_meta, batch_data in tqdm(data_loader):
            attentions, input_ids = get_attention(model, batch_meta, batch_data, device_id)
            # stack and rearrange to b, n_layers, n_heads, seq_len, seq_len
            attentions = torch.stack(attentions, dim=0)
            attentions = einops.rearrange(attentions, 'l b h x y -> b l h x y')
            attentions_lst.append(attentions)
            input_ids_lst.append(input_ids)
    
    return attentions_lst, input_ids_lst

def sample(attention_maps, input_ids, i, batch_size=8, vocab={}, punctuation=False, special_tokens=False): 
    remove_ids = [0] # 0 -> PAD
    if not (punctuation or special_tokens):
        if not punctuation:
            remove_ids.extend([vocab[c] for c in string.punctuation if c in vocab])
        if not special_tokens:
            remove_ids.extend([101, 102]) # 101 -> CLS, 102 -> SEP
    remove_ids = torch.tensor(remove_ids)
    mask = ~torch.isin(input_ids[i//batch_size][i%batch_size].cpu(), remove_ids)
    return attention_maps[i//batch_size][i%batch_size, :, :, mask, :][:, :, :, mask].cpu().numpy(), input_ids[i//batch_size][i%batch_size, mask].cpu().numpy()

def main():
    # eg. python attention_viz.py --finetuned_task coarse --finetuned_setting cross --compute_heads_importance --dont_normalize_importance_by_layer --dont_normalize_global_importance --devices 0 1 --seed 2022 --n_trials 10
    # parse args
    parser = argparse.ArgumentParser()
    parser = model_config(parser)
    parser = plot_config(parser)
    args = parser.parse_args()
    args.idx = [int(idx) for idx in args.idx] if args.idx else []
    args.devices = [int(d) for d in args.devices] if args.devices else []
    args.output_dir = os.path.join(args.output_dir, '_'.join([args.finetuned_task.upper(), args.finetuned_setting]))
    opt = vars(args)

    if args.compute_heads_importance:
        # Compute average head importances across multiple trials
        lst = []
        for i in range(args.n_trials):
            set_seed(args.seed-i)
            model, opt, test_data = create_model_and_dataloader(Experiment[args.finetuned_task.upper()], LingualSetting[args.finetuned_setting.upper()], opt, device_id=0, is_train=True)
            _, head_importance, _, preds, labels = compute_heads_importance(args, model, test_data)
            lst.append(head_importance)
        head_importance = torch.mean(torch.stack(lst), dim=0)
        head_ranks = torch.zeros(head_importance.numel(), dtype=torch.long, device=0)
        head_ranks[head_importance.view(-1).sort(descending=True)[1]] = torch.arange(
            head_importance.numel(), device=0
        )
        head_ranks = head_ranks.view_as(head_importance)

        os.makedirs(args.output_dir, exist_ok=True)
        np.save(os.path.join(args.output_dir, "head_importance.npy"), head_importance.detach().cpu().numpy())
        
        sns.set(font_scale=args.font_scale)
        plt.rcParams['figure.figsize'] = (args.figure_width, args.figure_height)
        plt.yticks(rotation=90)
        fig, axes = plt.subplots(1, 2)
        sns.heatmap(head_importance.cpu(), ax=axes[0], cmap="Reds", xticklabels=range(1, 12+1), yticklabels=range(1, 12+1), cbar=args.cbar, annot=True, fmt=".2f")
        axes[0].set_title('Head Importances')
        axes[0].set_xlabel('Head')
        axes[0].set_ylabel('Layer')
        sns.heatmap(head_ranks.cpu(), ax=axes[1], cmap="Reds_r", xticklabels=range(1, 12+1), yticklabels=range(1, 12+1), cbar=args.cbar, annot=True, fmt=".3g")
        axes[1].set_title('Head Ranks')
        axes[1].set_xlabel('Head')
        axes[1].set_ylabel('Layer')
        
        os.makedirs(args.output_dir, exist_ok=True)
        plt.savefig(os.path.join(args.output_dir, f'head_importance.png'))
    else:
        model, opt, test_data = create_model_and_dataloader(Experiment[args.finetuned_task.upper()], LingualSetting.BASE, opt, device_id=0, is_train=False)
        base_attentions, input_ids = get_attention_and_input_ids(model, test_data, device_id=0)
        model, opt, test_data = create_model_and_dataloader(Experiment[args.finetuned_task.upper()], LingualSetting[args.finetuned_setting.upper()], opt, device_id=1, is_train=False)
        finetuned_attentions, input_ids = get_attention_and_input_ids(model, test_data, device_id=1)

        tokenizer = BertTokenizer.from_pretrained(args.bert_model_type)

        layer, head = args.layer, args.head
        for idx in args.idx:
            ba, ip = sample(base_attentions, input_ids, idx, vocab=tokenizer.vocab, batch_size=args.batch_size)
            fa, ip = sample(finetuned_attentions, input_ids, idx, vocab=tokenizer.vocab, batch_size=args.batch_size)
            tokens = tokenizer.convert_ids_to_tokens(ip)

            sns.set(font_scale=args.font_scale)
            plt.rcParams['figure.figsize'] = (args.figure_width, args.figure_height)
            fig, axes = plt.subplots(1, 2)
            sns.heatmap(ba[layer-1, head-1], ax=axes[0], cmap="Blues", xticklabels=tokens, yticklabels=tokens, cbar=args.cbar)
            axes[0].set_title('Pre-trained')
            sns.heatmap(fa[layer-1, head-1], ax=axes[1], cmap="Blues", xticklabels=tokens, yticklabels=tokens, cbar=args.cbar)
            axes[1].set_title('Fine-tuned')
            fig.suptitle(f'Sample {idx}')
            
            os.makedirs(args.output_dir, exist_ok=True)
            plt.savefig(os.path.join(args.output_dir, f'layer_{layer}_head_{head}_sample_{idx}.png'))

if __name__ == '__main__':
    main()