from datasets import load_dataset, concatenate_datasets, Dataset
from pathlib import Path
import sys
from collections import OrderedDict
import numpy as np
import csv
sys.path.append('/home/june/mt-dnn/')
from experiments.exp_def import LingualSetting

def _prepare_data(train_langs, test_langs, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)

    for split in ['train', 'test']:
        out_file = out_dir.joinpath(f'marc_{split}_tmp.json')
        final_out_file = out_dir.joinpath(f'marc_{split}.tsv')

        if final_out_file.is_file():
            continue
        else:
            if not Path(out_file).is_file():
                datasets = []
                if split == 'train':
                    langs = train_langs
                else:
                    langs = test_langs
                
                if langs is None:
                    continue
                    
                for lang in langs:
                    dataset = load_dataset('amazon_reviews_multi', lang, split=split)
                    datasets.append(dataset)

                dataset = concatenate_datasets(datasets)
                dataset.to_json(out_file)

            # load and save as tsv in mtdnn format
            df = Dataset.from_json(str(out_file))
            with open(final_out_file, 'w') as f:
                for i, row in enumerate(df):
                    premise = row['review_body']
                    label = row['stars']
                    f.write(f'{i}\t{label}\t{premise}\n')

def cleave():
    df = Dataset.from_json('experiments/MARC/foreign/marc_test_tmp.json')
    final_out_files = [f'experiments/MARC/foreign_{i}/marc_test.tsv' for i in range(4)]
    fios = []
    for fof in final_out_files:
        Path(fof).parent.mkdir(parents=True, exist_ok=True)
        fios.append(open(fof, 'w'))

    for i, row in enumerate(df):
        premise = row['review_body']
        label = row['stars']
        if i < int(len(df) / 4):
            f = fios[0]
        elif i > int(len(df) / 4) and i <= int(len(df) / 4) * 2:
            f = fios[1]
        elif i > int(len(df) / 4) * 2 and i <= int(len(df) / 4) * 3:
            f = fios[2]
        else:
            f = fios[3]
        f.write(f'{i}\t{label}\t{premise}\n')
    
    for f in fios:
        f.close()

def prepare_finetune_data():
    train_langs_per_setting = {
        LingualSetting.CROSS: ['en'],
        LingualSetting.MULTI: ['en', 'fr', 'de', 'es']
    }
    test_langs = ['en', 'fr', 'de', 'es']

    for setting in [LingualSetting.CROSS, LingualSetting.MULTI]:
        out_dir = Path(f'experiments/MARC/{setting.name.lower()}')
        train_langs = train_langs_per_setting[setting]

        _prepare_data(train_langs, test_langs, out_dir)

def subsample_and_combine(foreign_dataset, ps):
    def read_rows(filename):
        with open(filename, 'r') as f:
            rows = []
            for row in f:
                id_, label, premise = row.split("\t")
                premise = premise.strip('\n')
                rows.append(OrderedDict({'id': id_, 'label': label, 'premise': premise}))
        return rows

    fieldnames = ['id', 'label', 'premise']
    mnli_rows = read_rows('experiments/MARC/cross/marc_train.tsv')

    seeds = [list(range(500, 900, 100)), list(range(900, 1300, 100)), list(range(1300, 1700, 100))]
    rows = read_rows(foreign_dataset)
    for i, seed_collection in enumerate(seeds):
        for p_idx, p in enumerate(ps):
            np.random.seed(seed_collection[p_idx])
            subsampled_idxs = np.random.choice(
                np.arange(len(rows)),
                size=int(len(rows)*p),
                replace=False)
            subsampled_rows = [rows[i] for i in subsampled_idxs]

            out_file = Path(f'experiments/MARC/foreign_{p}_{i}/marc_train.tsv')
            out_file.parent.mkdir(parents=True, exist_ok=True)

            with open(out_file, 'w') as fw:
                writer = csv.DictWriter(fw, fieldnames, delimiter='\t')
                for row in subsampled_rows:
                    writer.writerow(row)
            
                for r in mnli_rows:
                    writer.writerow(r)

if __name__ == '__main__':
    langs = ['es', 'fr', 'de']
    out_dir = Path(f'experiments/MARC/foreign')
    _prepare_data(langs, None, out_dir)
    foreign_dataset = 'experiments/MARC/foreign/marc_train.tsv'
    subsample_and_combine(foreign_dataset, [0.2, 0.4, 0.6, 0.8])

    # langs = ['es', 'fr', 'de']
    # out_dir = Path(f'experiments/MARC/foreign')
    # _prepare_data(None, langs, out_dir)
    # cleave()