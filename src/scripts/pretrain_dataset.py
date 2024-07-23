"""
Generate pretrain dataset for R2R and RxR
"""

import json
import jsonlines
import os.path as osp
from tqdm import tqdm
from transformers import BertTokenizer, XLMRobertaTokenizer

import sys
sys.path.append('src')
from dataset.R2R import R2R
from dataset.RxR import RxR
from config import R2R_DIR, RXR_DIR, BERT_TOKENIZER_DIR, XLM_ROBERTA_DIR


def prepare_view_indexes(dataset, split):
    view_indexes = dict()

    if dataset == 'R2R':
        dataset = R2R(split)
    if dataset == 'RxR':
        dataset = RxR(split)

    for batch, env in tqdm(dataset.get_batches(batch_size=1), total=len(dataset)):
        obs = env.reset()

        path = batch.at[0, 'path']
        path_id = batch.at[0, 'path_id']
        view_indexes[path_id] = []
        for step in range(len(path)):
            view_indexes[path_id].append(int(obs.at[0, 'view_index']))
            selection = obs.at[0, 'teacher_selection']
            obs = env.step_viewpoint(obs, [selection])
    return view_indexes


def prepare_dataset_jsonl(dataset, split):
    assert osp.exists(f'data/pretrain_v2')

    if dataset == 'R2R':
        if split == 'aug':
            path = osp.join(R2R_DIR, f'R2R_prevalent_aug.json')
        elif split == 'val_train':
            path = osp.join(R2R_DIR, f'R2R_val_train_seen.json')
        else:
            path = osp.join(R2R_DIR, f'R2R_{split}.json')
        with open(path) as f:
            data = json.load(f)

        view_indexes = prepare_view_indexes(dataset, split)
        tokenizer = BertTokenizer.from_pretrained(BERT_TOKENIZER_DIR)
        for i, item in enumerate(data):
            path_id = item['path_id']
            if isinstance(path_id, str):  # prevalent aug path_id looks like Prevalent_1234567_8
                data[i]['path_id'] = int(''.join(path_id.split('_')[-2:]))
            data[i]['path_view_index'] = view_indexes[path_id]
            data[i]['instructions_ids'] = tokenizer(item['instructions']).input_ids  # this contains mutiple instructions

    if dataset == 'RxR':
        if split == 'test':
            path = osp.join(RXR_DIR, f'rxr_test_standard_public_guide.jsonl')
        elif split == 'aug':
            path = osp.join(RXR_DIR, f'rxr_marky_train_guide.jsonl')
        else:
            path = osp.join(RXR_DIR, f'rxr_{split}_guide.jsonl')

        view_indexes = prepare_view_indexes(dataset, split)
        tokenizer = XLMRobertaTokenizer.from_pretrained(XLM_ROBERTA_DIR)
        with jsonlines.open(path) as f:
            data = []
            for item in f:
                sample = dict(item)
                if 'timed_instruction' in sample:
                    del sample['timed_instruction']
                if 'annotator_id' in sample:
                    del sample['annotator_id']
                if 'edit_distance' in sample:
                    del sample['edit_distance']

                path_id = item['path_id']
                sample['path_view_index'] = view_indexes[path_id]
                sample['instruction_ids'] = tokenizer(item['instruction']).input_ids  # there's only a single instruciton
                data.append(sample)


if __name__ == '__main__':
    for dataset in ['R2R', 'RxR']:
        for split in ['val_seen', 'val_unseen', 'train', 'aug']:
            # for the augment dataset, it will take about 2 hours and 13G memory on linux server.
            prepare_dataset_jsonl(dataset, split)


"""
# for CLIP tokenizer

from tqdm import tqdm
data = []
name = 'aug'
with jsonlines.open(f'data/pretrain_v2/R2RPretrain_{name}.jsonl') as f:
    print(f)
    for line in tqdm(f):
        line['instructions_ids'] = tok(line['instructions'], max_length=77).input_ids
        data.append(line)
with jsonlines.open(f'data/pretrain_v3/R2RPretrain_{name}.jsonl', mode='w') as f:
    f.write_all(data)
"""
