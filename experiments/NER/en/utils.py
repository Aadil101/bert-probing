from datasets import load_dataset, concatenate_datasets, Dataset
import pandas as pd
from pathlib import Path
import json

label_map = {
    'O': 0,
    'B-PER': 1,
    'I-PER': 2,
    'B-ORG': 3,
    'I-ORG': 4,
    'B-LOC': 5,
    'I-LOC': 6
}
label_map = {v:k for k,v in label_map.items()}

# en fr de es
for split in ['train', 'dev', 'test']:
    out_file = f'experiments/NER/en/ner_{split}_tmp.json'
    if not Path(out_file).is_file():
        datasets = []
        for lang in ['en']:
            dataset = load_dataset('wikiann', lang, split='validation' if split == 'dev' else split)
            datasets.append(dataset)

        dataset = concatenate_datasets(datasets)
        dataset.to_json(out_file)

    # load and save as tsv in mtdnn format
    df = Dataset.from_json(out_file)
    final_out_file = f'experiments/NER/en/ner_{split}.tsv'
    with open(final_out_file, 'w') as f:
        for i, row in enumerate(df):
            premise = ' '.join(row['tokens'])
            labels = ' '.join([label_map[int(ner_tag)] for ner_tag in row['ner_tags']])
            f.write(f'{i}\t{labels}\t{premise}\n')





