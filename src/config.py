R2R_DIR = './data/R2R'
RXR_DIR = './data/RxR'
CONNECTIVITY_DIR = './connectivity'


R2R_PRETRAIN_DIR = './data/pretrain_data'
RXR_PRETRAIN_DIR = './data/pretrain_data'


FEATURE_PATH = './data/img_features/ViT-patch16-224-CLIP.pkl'
# FEATURE_PATH = './data/img_features/ViT-patch16-384-ALBEF.pkl'
# FEATURE_PATH = 'data/img_features/dinov2-patch14-518.pkl'


CANDIDATE_BUFFER_PATH = './data/candidate_buffer.pkl'


BERT_DIR = '/home/lurenjie/documents/pretrained/bert-base-uncased'
BERT_TOKENIZER_DIR = '/home/lurenjie/documents/pretrained/bert-base-uncased-tokenizer'
XLM_ROBERTA_DIR = '/home/lurenjie/documents/pretrained/xlm-roberta-base'
CLIP_DIR = '/home/lurenjie/documents/pretrained/clip-vit-base-patch16'
ALBEF_PATH = './data/pretrained_model/ALBEF_refCOCO+.pt'


import torch
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
