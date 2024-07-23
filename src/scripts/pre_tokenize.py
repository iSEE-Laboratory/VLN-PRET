from transformers import BertTokenizer

DIR = '/home/lurenjie/documents/pretrained/bert-base-uncased-tokenizer'
tokenizer = BertTokenizer.from_pretrained(DIR)


import os
import json
import pickle
import os.path as osp
R2R_DIR = 'data/R2R'
CACHE_DIR = osp.join(R2R_DIR, 'encoded_cache_dir')
os.makedirs(CACHE_DIR, exist_ok=True)
for split in ['train', 'prevalent_aug', 'val_train_seen', 'val_seen', 'val_unseen', 'test']:
    # read data
    path = osp.join(R2R_DIR, f'R2R_{split}.json')

    # process
    with open(path) as f:
        dataset = json.load(f)
    for item in dataset:
        # tokenize on list[str]
        item['instructions'] = tokenizer(item['instructions']).input_ids

    # save
    path = osp.join(CACHE_DIR, f'R2R_{split}.pkl')
    with open(path, 'wb') as f:
        pickle.dump(dataset, f)
