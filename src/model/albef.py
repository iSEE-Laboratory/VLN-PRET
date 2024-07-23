"""
code from ALBEF. I modified it for easy use.
"""

import torch
from torch import nn
from functools import partial
from torchvision import transforms
from timm.models.vision_transformer import VisionTransformer
from transformers import BertConfig, BertModel, BertTokenizer, PretrainedConfig


class Identity(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, inputs_embeds, *args, **kwargs):
        return inputs_embeds


class ALBEF(nn.Module):
    def __init__(self, bert_tokenizer_dir='bert-base-uncased'):
        super().__init__()

        self.bert_tokenizer_dir = bert_tokenizer_dir
        self.tokenizer = BertTokenizer.from_pretrained(self.bert_tokenizer_dir)

        # ViT
        self.visual_encoder = VisionTransformer(
            img_size=384, patch_size=16, embed_dim=768, depth=12, num_heads=12, 
            mlp_ratio=4, qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6)) 
        self.visual_encoder.reset_classifier(num_classes=0)  # remove classification head

        # 6 layer transformer encoder and embedding layer
        bert_config = BertConfig(num_hidden_layers=6)
        self.text_encoder = BertModel(bert_config, add_pooling_layer=False)
        # self.text_encoder.encoder.gradient_checkpointing = True

        # 6 layer transformer decoder, used to predict MLM
        bert_config = BertConfig(num_hidden_layers=6, is_decoder=True, add_cross_attention=True)
        self.fusion_module = BertModel(bert_config, add_pooling_layer=False)
        self.fusion_module.embeddings = Identity()  # remove the embedding layer, simply delete it will raise error.

        self.itm_head = nn.Linear(768, 2)

    def get_tokenizer(self):
        return self.tokenizer

    def get_transform(self):
        transform = transforms.Compose([
            # transforms.Resize((384, 384),interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])
        return transform

    def forward_text(self, input_ids, attention_mask):
        outputs = self.text_encoder(
            input_ids,
            attention_mask=attention_mask,
            return_dict=True,
        )
        return outputs.last_hidden_state

    def forward_image(self, image):
        image_tokens = self.visual_encoder.forward_features(image)
        return image_tokens

    def forward_fusion(self, key_tokens, value_tokens, attention_mask, value_padding_mask=None, output_attentions=False):
        """
        Args:
            attention_mask: (B, N), indicate which elements are valid.
            cross_attention_mask: (B, N)
        """
        assert len(attention_mask.shape) == 2
        # decoder默认会有causal mask
        # 把mask变为3维就不会有了
        attention_mask = attention_mask.unsqueeze(1)

        outputs = self.fusion_module(
            inputs_embeds=key_tokens,
            attention_mask=attention_mask,
            encoder_hidden_states=value_tokens,
            encoder_attention_mask=value_padding_mask,
            output_attentions=output_attentions,
        )
        return outputs

    def forward(self, image, input_ids, attention_mask, output_attentions=False):
        image_tokens = self.forward_image(image)
        text_tokens = self.forward_text(input_ids, attention_mask)
        outputs = self.forward_fusion(image_tokens, text_tokens, attention_mask, output_attentions)
        
        cls = outputs.last_hidden_state[:, 0 ,:]
        itm = self.itm_head(cls)   
        return outputs, itm
