from datasets import load_dataset, concatenate_datasets, Dataset
import pandas as pd
from pathlib import Path
import json

# CoNLL-2013 Dataset

label_map = {
    '"': 0,
     "''": 1, 
    '#': 2, 
    '$': 3, 
    '(': 4, 
    ')': 5, 
    ',': 6, 
    '.': 7, 
    ':': 8, 
    '``': 9, 
    'CC': 10, 
    'CD': 11, 
    'DT': 12,
    'EX': 13, 
    'FW': 14, 
    'IN': 15, 
    'JJ': 16, 
    'JJR': 17, 
    'JJS': 18, 
    'LS': 19, 
    'MD': 20, 
    'NN': 21, 
    'NNP': 22, 
    'NNPS': 23,
    'NNS': 24, 
    'NN|SYM': 25, 
    'PDT': 26, 
    'POS': 27, 
    'PRP': 28, 
    'PRP$': 29, 
    'RB': 30, 
    'RBR': 31, 
    'RBS': 32, 
    'RP': 33,
    'SYM': 34, 
    'TO': 35, 
    'UH': 36, 
    'VB': 37, 
    'VBD': 38, 
    'VBG': 39, 
    'VBN': 40, 
    'VBP': 41, 
    'VBZ': 42, 
    'WDT': 43,
    'WP': 44, 
    'WP$': 45, 
    'WRB': 46
}

label_map = {v:k for k,v in label_map.items()}

# en fr de es
for split in ['train', 'dev', 'test']:
    out_file = f'experiments/POS/en/pos_{split}_tmp.json'
    if not Path(out_file).is_file():
        datasets = []
        for lang in ['en']:
            dataset = load_dataset('conll2003', split='validation' if split == 'dev' else split)
            datasets.append(dataset)

        dataset = concatenate_datasets(datasets)
        dataset.to_json(out_file)

    # load and save as tsv in mtdnn format
    df = Dataset.from_json(out_file)
    final_out_file = f'experiments/POS/en/pos_{split}.tsv'
    with open(final_out_file, 'w') as f:
        for i, row in enumerate(df):
            premise = ' '.join(row['tokens'])
            labels = ' '.join([label_map[int(pos_tag)] for pos_tag in row['pos_tags']])
            f.write(f'{i}\t{labels}\t{premise}\n')





