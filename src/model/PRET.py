import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForMaskedLM, CLIPModel, CLIPProcessor, BertModel

import utils
from config import BERT_TOKENIZER_DIR, BERT_DIR, ALBEF_PATH, XLM_ROBERTA_DIR, CLIP_DIR, device

from .albef import ALBEF


class PRET(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.text_backbone_name = config.text_backbone
        self.multilingual = (config.dataset == 'RxR' or config.dataset == 'RxRPretrain')
        self.dropout = config.dropout
        self.hidden_dim = 768
        self.nhead = 12
        self.OPE_layer_num = config.OPE_layer_num
        self.MAM_layer_num = config.MAM_layer_num
        self.CCM_layer_num = config.CCM_layer_num

        self.dropout = nn.Dropout(self.dropout)

        # text encoder
        self.text_model, self.text_max_length = get_text_model(self.text_backbone_name, self.multilingual)
        self.tokenizer = self.text_model.get_tokenizer()

        # OPE
        # It is strange that dropout in transformer is not reproducible even with fixed seed
        # I use pytorch 1.13 and cuda 11.8
        self.angle_embedding = nn.Linear(4, self.hidden_dim)
        self.vision_embedding = nn.Linear(768 + 128, self.hidden_dim)

        self.use_panorama = config.use_panorama
        self.use_directed = config.use_directed  # if not use panorama, this is ignored
        if self.use_panorama:
            if self.use_directed:
                self.OPE = get_transformer_decoder(self.hidden_dim, self.nhead, self.OPE_layer_num)
            else:
                self.OPE = get_transformer_encoder(self.hidden_dim, self.nhead, self.OPE_layer_num)

        # MAM
        self.MAM = get_transformer_decoder(self.hidden_dim, self.nhead, self.MAM_layer_num)
        self.cls = nn.Parameter(torch.zeros(1, 1, self.hidden_dim))

        # CCM
        self.CCM = get_transformer_encoder(self.hidden_dim, self.nhead, self.CCM_layer_num)
        self.mlp = nn.Sequential(
            nn.Linear(self.hidden_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )

    def forward_text(self, instructions):
        """
        Args:
            instructions: list[str] or list[list[int]]
        """
        assert isinstance(instructions, list)

        if isinstance(instructions[0], str):
            inputs = self.tokenizer(instructions,
                                    padding=True,
                                    return_tensors='pt',
                                    truncation=True,
                                    max_length=self.text_max_length).to(device=device)
        else:  # for encoded text
            inputs = self.tokenizer.batch_encode_plus(instructions,
                                                      add_special_tokens=False,
                                                      is_split_into_words=True,
                                                      padding=True,
                                                      return_tensors='pt',
                                                      truncation=True,
                                                      max_length=self.text_max_length).to(device)
        self.tokens = self.tokenizer.convert_ids_to_tokens(inputs.input_ids[0])  # use for visualization

        text_features = self.text_model.forward_text(inputs.input_ids, inputs.attention_mask)
        text_padding_mask = ~inputs.attention_mask.bool()
        return text_features, text_padding_mask

    def forward_OPE(self, candidate_feature, panoramic_feature, padding_mask):
        """
        Args:
            angle_features: (B, candidate_num, 128 + 768) or (B, candidate_num, 4)
            panoramic_features: (B, 36, 128 + 768)
            padding_mask: (B, candidate_num)
        """
        if self.use_panorama:
            panoramic_feature = self.dropout(panoramic_feature)
            panoramic_feature = self.vision_embedding(panoramic_feature)  # （B, 36, 768)

            if self.use_directed:
                candidate_feature = candidate_feature[:, :, :4]  # take the angle feature only
                candidate_feature = self.angle_embedding(candidate_feature)  # (B, candidate_num, 768)
                tokens = self.OPE(candidate_feature, panoramic_feature, tgt_key_padding_mask=padding_mask)  # (B, candidate_num, 768)
            else:
                tokens = self.OPE(panoramic_feature)  # (B, 36, 768)
        else:
            candidate_feature = self.dropout(candidate_feature)
            tokens = self.vision_embedding(candidate_feature)  # (B, candidate_num, 768)
        return tokens

    def forward_MAM(self,
            text_features,
            text_padding_mask,
            path_features,
            path_padding_mask,
            local_features=None,
            local_padding_mask=None,):
        B, N, C = path_features.shape
        path_only = local_features is None
        if path_only:
            local_features = torch.zeros(B, 0, C, device=path_features.device)
            local_padding_mask = torch.zeros(B, 0, dtype=torch.bool, device=path_features.device)

        assert text_padding_mask.dtype == torch.bool
        assert path_padding_mask.dtype == torch.bool
        assert local_padding_mask.dtype == torch.bool

        B, M, C = local_features.shape
        pos = torch.arange(N, device=path_features.device)
        pe = position_embedding(pos, dim=C)
        path_features = path_features + pe

        pe = position_embedding(torch.tensor([N], device=local_features.device), dim=C)
        local_features[:, 1:, :] = local_features[:, 1:, :] + pe  # add position embedding except STOP

        # prepare inputs, cls is used as start token
        cls = self.cls.expand(B, 1, -1)
        falses = torch.zeros(B, cls.shape[1], dtype=torch.bool, device=cls.device)
        tokens = torch.cat([cls, path_features, local_features], dim=1)
        padding_mask = torch.cat([falses, path_padding_mask, local_padding_mask], dim=1)

        # prepare merged causal mask
        mask = torch.zeros(tokens.shape[1], tokens.shape[1], dtype=torch.bool, device=device)
        i = N + cls.shape[1]  # +1 for cls
        j = i + M
        mask[i:, :] = True
        mask[:i, :i] = utils.get_causal_mask(i).to(tokens.device)
        mask[i:j, i:j] = torch.eye(local_features.shape[1], dtype=torch.bool, device=device)
        mask = ~mask

        # forward
        tokens = self.MAM(
            tokens, text_features,
            tgt_mask=mask,
            tgt_key_padding_mask=padding_mask,
            memory_key_padding_mask=text_padding_mask)

        # return
        cls, path_tokens, local_tokens = torch.split(tokens, [cls.shape[1], N, M], dim=1)
        if path_only:
            return path_tokens
        else:
            return path_tokens, local_tokens

    def forward_CCM(self, path_feature, padding_mask=None):
        """
        Args:
            path_feature: (B, N, C)
            padding_mask: (B, N)
        Returns:
            (B, N)
        """
        path_feature = self.CCM(path_feature, src_key_padding_mask=padding_mask)
        path_score = self.mlp(path_feature).squeeze(2)
        path_score = path_score + padding_mask * -10000.0
        return path_score


def get_text_model(model_name, multilingual):
    """
    I use the ALBEF as the base model, supporting different text model by replacing the tokenizer and backbone in it.
    # ALBEF, 6 layer, 768
    # CLIP, 12 layer, 512
    # BERT, 12 layer, 768
    """
    assert model_name in {'ALBEF', 'CLIP', 'BERT'}

    text_model = ALBEF(BERT_TOKENIZER_DIR)

    if multilingual:
        text_max_length = 512
        print('Loading RoBERTa...')
        tokenizer = AutoTokenizer.from_pretrained(XLM_ROBERTA_DIR)
        roberta = AutoModelForMaskedLM.from_pretrained(XLM_ROBERTA_DIR).roberta
        roberta.encoder.gradient_checkpointing = True  # do not use gradient checkpoint when enable DDP
        text_model.tokenizer = tokenizer
        text_model.text_encoder = roberta
    elif model_name == 'ALBEF':
        text_max_length = 512
        print('Loading ALBEF...')
        checkpoint = torch.load(ALBEF_PATH, map_location='cpu')
        text_model.load_state_dict(checkpoint)
    elif model_name == 'CLIP':
        print('Loading CLIP...')
        text_max_length = 77
        clip = CLIPModel.from_pretrained(CLIP_DIR)
        text_model.tokenizer = CLIPProcessor.from_pretrained(CLIP_DIR).tokenizer
        text_model.text_encoder = clip.text_model
    elif model_name == 'BERT':
        print('Loading BERT...')
        text_model.text_encoder = BertModel.from_pretrained(BERT_DIR)

    del text_model.visual_encoder

    return text_model, text_max_length


def get_transformer_encoder(hidden_dim, nhead, layer_num):
    encoder_layer = nn.TransformerEncoderLayer(
        hidden_dim,
        nhead=nhead,
        dim_feedforward=hidden_dim * 4,
        activation=F.gelu,
        batch_first=True)
    return nn.TransformerEncoder(encoder_layer, num_layers=layer_num)


def get_transformer_decoder(hidden_dim, nhead, layer_num):
    decoder_layer = nn.TransformerDecoderLayer(
        d_model=hidden_dim,
        nhead=nhead,
        dim_feedforward=hidden_dim * 4,
        activation=F.gelu,
        batch_first=True
    )
    return nn.TransformerDecoder(decoder_layer, num_layers=layer_num)


def position_embedding(pos, dim=768, max_len=10000):
    """
    Args:
        pos: tensor
        dim: dimension of the token, normally it is 768.
        device: create the tensor on cpu or gpu.
    Returns:
        position embedding, a (dim,) tensor
    """
    x = torch.arange(1, dim + 1, device=pos.device)
    phi = (x % 2 == 0) * (torch.pi / 2)

    x[x % 2 == 1] += 1
    if isinstance(pos, torch.Tensor):
        for i in range(len(pos.shape)):
            x.unsqueeze(0)
        x = pos.unsqueeze(-1) / (max_len ** (x / dim))
    else:
        x = pos / (max_len ** (x / dim))

    pe = torch.sin(x + phi)
    return pe
