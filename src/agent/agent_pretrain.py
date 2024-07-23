import torch
import random
import os.path as osp
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

import utils
from config import device


class AgentPretrain(nn.Module):
    def __init__(self, config, model):
        super().__init__()

        hidden_dim = model.hidden_dim

        self.model = model
        self.tok = model.tokenizer

        self.mask_visited = config.mask_visited
        self.use_directed = config.use_directed
        self.use_panorama = config.use_panorama

        self.tasks = config.tasks

        assert 'MLM' in self.tasks  # at least one pretraining task.
        self.mlm_head = nn.Linear(768, self.tok.vocab_size)

        # NOTE: the ITM does not help at all, I don't know why
        if 'ITM' in self.tasks:
            self.itm_head = nn.Linear(hidden_dim, 1)

    def forward(self, dataset_name, batch):
        if dataset_name == 'MLM':
            path_id, text_ids, history_panorama, history_candidates_list, history_teacher_selection_list, history_length = batch
            path_id = path_id.cuda()
            text_ids = text_ids.cuda()
            history, history_padding_mask = self.forward_history(history_panorama, history_candidates_list, history_teacher_selection_list, history_length)
            if self.training:
                loss = self._forward_MLM(text_ids, history, history_padding_mask)
                loss_log = {'MLM': loss.item()}
                return loss, loss_log
            else:
                prediction_MLM, target_MLM = self._forward_MLM(text_ids, history, history_padding_mask)
                return prediction_MLM, target_MLM

        elif dataset_name == 'ITM':
            path_id, text_ids, history_panorama, history_candidates_list, history_teacher_selection_list, history_length = batch
            path_id = path_id.cuda()
            text_ids = text_ids.cuda()
            history, history_padding_mask = self.forward_history(history_panorama, history_candidates_list, history_teacher_selection_list, history_length)
            if self.training:
                loss = self._forward_ITM(path_id, text_ids, history, history_padding_mask)
                loss_log = {'ITM': loss.item()}
                return loss, loss_log
            else:
                prediction_ITM, target_ITM = self._forward_ITM(path_id, text_ids, history, history_padding_mask)
                return prediction_ITM, target_ITM

        else:
            raise RuntimeError('unknown task.')

    def forward_history(self, history_panorama, history_candidates_list, history_teacher_selection_list, history_length):
        """
        Args:
            history_panorama: (B * history_length, 36, 768)
            history_candidates_list: list[list[Tensor(candidate_num, 768 + 128)]]
            history_teacher_seleciton_list: list[Tensor(B,)]
            history_length: Tensor(B,)
        Returns:
            history_feature: Tensor(B, history_length, 768)
            padding_mask: Tensor(B, history_length)
        """
        history_panorama = history_panorama.cuda()

        # prepare candidate feature
        all_candidate_num = []
        all_candidates = []
        for i, history_candidates in enumerate(history_candidates_list):
            all_candidate_num.extend([candidates.shape[0] for candidates in history_candidates])
            all_candidates.extend(history_candidates)
        all_candidate_features = pad_sequence(all_candidates, batch_first=True).cuda()  # (B * history_length, max_candidate_num, 768)
        padding_mask = utils.length_to_mask(all_candidate_num, device=all_candidate_features.device)  # (B * history_length, max_candidate_num)

        # forward OPE
        all_candidate_features = self.model.forward_OPE(all_candidate_features, history_panorama, padding_mask)
        if self.use_directed:
            all_candidate_features = F.pad(all_candidate_features, (0, 0, 1, 0))  # add STOP embedding, 方便index选择
        batch_candidate_features = torch.split(all_candidate_features, history_length.tolist(), dim=0)  # list[Tensor(history_length, max_candidate_num, 768)]

        # gather history features.
        history_features = []
        for i, history_teacher_selection in enumerate(history_teacher_selection_list):
            # history_teacher_selection的最后一个肯定是0，即停止行为
            history_candidate_features = batch_candidate_features[i]  # (history_length, max_candidate_num, 768)
            if self.use_directed:
                # 如果history_teacher_selection为0，说明history length为 0
                # 这里的index是加了1的，0表示停止行为，所以前面要加上STOP embedding
                index = history_teacher_selection.cuda().view(-1, 1, 1).expand(-1, 1, history_candidate_features.shape[-1])
                history_feature = history_candidate_features.gather(dim=1, index=index).squeeze(1)  # (history_length, 768)
            else:
                history_feature = history_candidate_features.mean(dim=1)  # average of 36 image feature
                history_feature[-1, :] = 0  # the last token is the stop token
            history_features.append(history_feature)
        history_features = pad_sequence(history_features, batch_first=True)
        padding_mask = utils.length_to_mask(history_length.tolist(), device=history_features.device)
        return history_features, padding_mask

    def _forward_MLM(self, text_ids, history, history_padding_mask):
        """
        Args:
            text_ids: (B, L)
            history: (B, N, C)
            history_length: (B, N)
        """
        attention_mask = (text_ids != 0).float()
        masked_text_ids, masked_mask, masked_ids = utils.MLM(
            text_ids, attention_mask,
            sep_token_id=self.tok.sep_token_id,
            mask_token_id=self.tok.mask_token_id,
            vocab_size=self.tok.vocab_size)
        text_features = self.model.text_model.forward_text(masked_text_ids, attention_mask)

        history_mask = (~history_padding_mask).float()
        tokens = self.model.text_model.forward_fusion(text_features, history, attention_mask, history_mask).last_hidden_state

        logits = self.mlm_head(tokens[masked_mask])  # (masked_num, vocab_size)
        if self.training:
            loss = F.cross_entropy(logits, masked_ids)
            return loss
        else:
            return logits.argmax(dim=1), masked_ids

    def _forward_ITM(self, path_id, text_ids, history, history_padding_mask):
        B, _, C = history.shape

        # positive samples
        # history里多包含了一步，需要删掉，并且把最后一个替换为stop_embedding
        positive_sample = history
        positive_mask = history_padding_mask

        # 负样本
        # replacing with other history in the batch
        k = 3
        negative_samples = []
        negative_masks = []
        def random_sample(seq, exclude):
            choices = [i for i in seq if path_id[i] != exclude]
            assert len(choices) >= k
            return random.sample(choices, k=k)
        replace_idx_list = [random_sample(range(B), path_id[i]) for i in range(B)]  # list[list[int]]
        for replace_idx in zip(*replace_idx_list):
            negative_samples.append(positive_sample[replace_idx, :, :])
            negative_masks.append(positive_mask[replace_idx, :])

        # shuffle
        max_length = history_padding_mask.shape[1]
        history_lengths = max_length - history_padding_mask.sum(dim=1)

        negative_sample_2 = history[:]
        negative_mask_2 = history_padding_mask[:]
        for i in range(B):
            length = history_lengths[i]
            shuffle_idx = list(range(length))
            random.shuffle(shuffle_idx)
            shuffle_idx = shuffle_idx + list(range(length, max_length))
            negative_sample_2[i] = negative_sample_2[i, shuffle_idx, :]
        negative_samples.append(negative_sample_2)
        negative_masks.append(negative_mask_2)

        # forward text
        attention_mask = (text_ids != 0).float()
        text_features = self.model.text_model.forward_text(text_ids, attention_mask)
        text_features = text_features.repeat(1 + len(negative_samples), 1, 1)
        attention_mask = attention_mask.repeat(1 + len(negative_samples), 1)

        # forward ITM
        tokens = torch.cat([positive_sample] + negative_samples, dim=0)
        padding_mask = torch.cat([positive_mask] + negative_masks, dim=0)
        path_tokens = self.model.forward_MAM(
            text_features, (1 - attention_mask).bool(),
            tokens, padding_mask)

        # loss
        last_token_index = padding_mask.shape[1] - padding_mask.sum(dim=1) - 1
        last_token_index = last_token_index.view(-1, 1, 1).expand(-1, 1, C)
        final_tokens = path_tokens.gather(index=last_token_index, dim=1).squeeze(1)  # (N * B, C)
        logits = self.itm_head(final_tokens)  # (N * B, 1)
        logits = logits.reshape(1 + len(negative_samples), B).permute(1, 0)
        target = torch.zeros(B, dtype=torch.long, device=logits.device)
        if self.training:
            loss = F.cross_entropy(logits, target)
            return loss
        else:
            return logits.argmax(dim=1), target  # used to calculate accuracy

    def save(self, log_dir):
        for name, model in self.named_children():
            path = osp.join(log_dir, f'{name}.pt')
            torch.save(model.state_dict(), path)

    def load(self, log_dir, strict=True):
        load_any = False
        for name, model in self.named_children():
            path = osp.join(log_dir, f'{name}.pt')
            if osp.exists(path):
                state_dict = torch.load(path, map_location='cpu')
                model.load_state_dict(state_dict)
                print(f'{path} loaded!')
                load_any = True
        assert load_any
