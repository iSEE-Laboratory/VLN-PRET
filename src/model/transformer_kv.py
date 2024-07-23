"""
Implement KV Cache for nn.TransformerDecoder
"""

import copy
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class TransformerKVDecoder(nn.Module):
    def __init__(self, decoder_layer, num_layers):
        super().__init__()
        self.layers = nn.ModuleList([copy.deepcopy(decoder_layer) for i in range(num_layers)])
        self.num_layers = num_layers

    def forward(self, tgt, memory, tgt_mask=None,
                memory_mask=None, tgt_key_padding_mask=None,
                memory_key_padding_mask=None,
                kv_cache=None, get_cache=False):
        output = tgt

        key_values = []
        for i, mod in enumerate(self.layers):
            if get_cache:
                output, layer_key_values = mod(output, memory, tgt_mask=tgt_mask,
                    memory_mask=memory_mask,
                    tgt_key_padding_mask=tgt_key_padding_mask,
                    memory_key_padding_mask=memory_key_padding_mask,
                    get_cache=True)
                key_values.append(layer_key_values)
            elif kv_cache is not None:
                layer_kv_cache = kv_cache[i]
                output, layer_key_values = mod(output, memory, tgt_mask=tgt_mask,
                    memory_mask=memory_mask,
                    tgt_key_padding_mask=tgt_key_padding_mask,
                    memory_key_padding_mask=memory_key_padding_mask,
                    kv_cache=layer_kv_cache)
                key_values.append(layer_key_values)
            else:
                output = mod(output, memory, tgt_mask=tgt_mask,
                    memory_mask=memory_mask,
                    tgt_key_padding_mask=tgt_key_padding_mask,
                    memory_key_padding_mask=memory_key_padding_mask)
        return output, key_values


class TransformerKVDecoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, dropout=0.1,
                 activation=F.relu,
                 layer_norm_eps=1e-5, batch_first=False,
                 device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first,
                                            **factory_kwargs)
        self.multihead_attn = MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first,
                                                 **factory_kwargs)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward, **factory_kwargs)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model, **factory_kwargs)

        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.norm3 = nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        # Legacy string support for activation function.
        if isinstance(activation, str):
            if activation == "relu":
                self.activation = F.relu
            elif activation == "gelu":
                self.activation = F.gelu
        else:
            self.activation = activation

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None,
                tgt_key_padding_mask=None, memory_key_padding_mask=None,
                kv_cache=None, get_cache=False):
        x = tgt

        if get_cache:
            residual, sa_key_values = self._sa_block(x, tgt_mask, tgt_key_padding_mask, get_cache=True)
            x = self.norm1(x + residual)
            residual, mha_key_values = self._mha_block(x, memory, memory_mask, memory_key_padding_mask, get_cache=True)
            x = self.norm2(x + residual)
            x = self.norm3(x + self._ff_block(x))
            return x, sa_key_values + mha_key_values

        elif kv_cache is not None:
            sa_kv_cache = kv_cache[:2]
            mha_kv_cache = kv_cache[2:]
            residual, sa_key_values = self._sa_block(x, tgt_mask, tgt_key_padding_mask, kv_cache=sa_kv_cache)
            x = self.norm1(x + residual)
            residual, mha_key_values = self._mha_block(x, memory, memory_mask, memory_key_padding_mask, kv_cache=mha_kv_cache)
            x = self.norm2(x + residual)
            x = self.norm3(x + self._ff_block(x))
            return x, sa_key_values + mha_key_values

        else:
            x = self.norm1(x + self._sa_block(x, tgt_mask, tgt_key_padding_mask))
            x = self.norm2(x + self._mha_block(x, memory, memory_mask, memory_key_padding_mask))
            x = self.norm3(x + self._ff_block(x))
            return x

    def _sa_block(self, x,
                  attn_mask, key_padding_mask,
                  kv_cache=None, get_cache=False):

        if get_cache:
            x, key_values = self.self_attn(x, x, x,
                attn_mask=attn_mask,
                key_padding_mask=key_padding_mask,
                need_weights=False,
                get_cache=True)
            return self.dropout1(x), key_values

        elif kv_cache is not None:
            x, key_values = self.self_attn(x, x, x,
                attn_mask=attn_mask,
                key_padding_mask=key_padding_mask,
                need_weights=False,
                kv_cache=kv_cache)
            return self.dropout1(x), key_values

        else:
            x = self.self_attn(x, x, x,
                attn_mask=attn_mask,
                key_padding_mask=key_padding_mask,
                need_weights=False)[0]
            return self.dropout1(x)

    def _mha_block(self, x, mem,
                   attn_mask, key_padding_mask,
                   kv_cache=None, get_cache=False):

        if get_cache:
            x, key_values = self.multihead_attn(x, mem, mem,
                attn_mask=attn_mask,
                key_padding_mask=key_padding_mask,
                need_weights=False,
                get_cache=True)
            return self.dropout2(x), key_values

        elif kv_cache is not None:
            x, key_values = self.multihead_attn(x, mem, mem,
                attn_mask=attn_mask,
                key_padding_mask=key_padding_mask,
                need_weights=False,
                kv_cache=kv_cache)
            return self.dropout2(x), key_values

        else:
            x = self.multihead_attn(x, mem, mem,
                                    attn_mask=attn_mask,
                                    key_padding_mask=key_padding_mask,
                                    need_weights=False)[0]
            return self.dropout2(x)

    def _ff_block(self, x):
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout3(x)


class MultiheadAttention(nn.Module):

    def __init__(self, embed_dim, num_heads, dropout=0., bias=True, add_bias_kv=False,
                 kdim=None, vdim=None, batch_first=False, device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(MultiheadAttention, self).__init__()
        self.embed_dim = embed_dim
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        self._qkv_same_embed_dim = self.kdim == embed_dim and self.vdim == embed_dim
        assert self._qkv_same_embed_dim

        self.num_heads = num_heads
        self.dropout = dropout
        self.batch_first = batch_first
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"

        self.in_proj_weight = nn.Parameter(torch.empty((3 * embed_dim, embed_dim), **factory_kwargs))
        self.register_parameter('q_proj_weight', None)
        self.register_parameter('k_proj_weight', None)
        self.register_parameter('v_proj_weight', None)

        if bias:
            self.in_proj_bias = nn.Parameter(torch.empty(3 * embed_dim, **factory_kwargs))
        else:
            self.register_parameter('in_proj_bias', None)
        self.out_proj = NonDynamicallyQuantizableLinear(embed_dim, embed_dim, bias=bias, **factory_kwargs)

        if add_bias_kv:
            self.bias_k = nn.Parameter(torch.empty((1, 1, embed_dim), **factory_kwargs))
            self.bias_v = nn.Parameter(torch.empty((1, 1, embed_dim), **factory_kwargs))
        else:
            self.bias_k = self.bias_v = None

        self.add_zero_attn = False

        self._reset_parameters()

    def _reset_parameters(self):
        if self._qkv_same_embed_dim:
            nn.init.xavier_uniform_(self.in_proj_weight)
        else:
            nn.init.xavier_uniform_(self.q_proj_weight)
            nn.init.xavier_uniform_(self.k_proj_weight)
            nn.init.xavier_uniform_(self.v_proj_weight)

        if self.in_proj_bias is not None:
            nn.init.constant_(self.in_proj_bias, 0.)
            nn.init.constant_(self.out_proj.bias, 0.)
        if self.bias_k is not None:
            nn.init.xavier_normal_(self.bias_k)
        if self.bias_v is not None:
            nn.init.xavier_normal_(self.bias_v)

    def forward(self, query, key, value, key_padding_mask=None,
                need_weights=True, attn_mask=None,
                average_attn_weights=True,
                kv_cache=None, get_cache=False):
        is_batched = query.dim() == 3
        if key_padding_mask is not None:
            _kpm_dtype = key_padding_mask.dtype
            if _kpm_dtype != torch.bool and not torch.is_floating_point(key_padding_mask):
                raise AssertionError(
                    "only bool and floating types of key_padding_mask are supported")

        if self.batch_first and is_batched:
            # make sure that the transpose op does not affect the "is" property
            if key is value:
                if query is key:
                    query = key = value = query.transpose(1, 0)
                else:
                    query, key = [x.transpose(1, 0) for x in (query, key)]
                    value = key
            else:
                query, key, value = [x.transpose(1, 0) for x in (query, key, value)]

        if get_cache:
            attn_output, key_values = forward_attention(
                query, key, value, self.embed_dim, self.num_heads,
                self.in_proj_weight, self.in_proj_bias,
                self.bias_k, self.bias_v, self.add_zero_attn,
                self.dropout, self.out_proj.weight, self.out_proj.bias,
                training=self.training,
                key_padding_mask=key_padding_mask, need_weights=need_weights,
                attn_mask=attn_mask, average_attn_weights=average_attn_weights)
        elif kv_cache is not None:
            attn_output, key_values = kv_cache_attention(
                query, key, value, kv_cache[0], kv_cache[1], self.num_heads,
                self.in_proj_weight, self.in_proj_bias,
                self.dropout, self.out_proj.weight, self.out_proj.bias,
                training=self.training,
                key_padding_mask=key_padding_mask,
                attn_mask=attn_mask,)
        else:
            attn_output, attn_output_weights = F.multi_head_attention_forward(
                query, key, value, self.embed_dim, self.num_heads,
                self.in_proj_weight, self.in_proj_bias,
                self.bias_k, self.bias_v, self.add_zero_attn,
                self.dropout, self.out_proj.weight, self.out_proj.bias,
                training=self.training,
                key_padding_mask=key_padding_mask, need_weights=need_weights,
                attn_mask=attn_mask, average_attn_weights=average_attn_weights)
            key_values = None
        if self.batch_first and is_batched:
            return attn_output.transpose(1, 0), key_values
        else:
            return attn_output, key_values


class NonDynamicallyQuantizableLinear(nn.Linear):
    def __init__(self, in_features: int, out_features: int, bias: bool = True,
                 device=None, dtype=None) -> None:
        super().__init__(in_features, out_features, bias=bias,
                         device=device, dtype=dtype)


def kv_cache_attention(
    query,
    key,
    value,
    k_cache,
    v_cache,
    num_heads: int,
    in_proj_weight,
    in_proj_bias,
    dropout_p,
    out_proj_weight,
    out_proj_bias,
    training=True,
    key_padding_mask=None,
    attn_mask=None,
):
    # set up shape vars
    tgt_len, bsz, embed_dim = query.shape
    bsz, _, src_len, _ = k_cache.shape
    head_dim = embed_dim // num_heads
    assert head_dim * num_heads == embed_dim, f"embed_dim {embed_dim} not divisible by num_heads {num_heads}"

    # compute in-projection, forward query only
    assert in_proj_weight is not None, "use_separate_proj_weight is False but in_proj_weight is None"
    q, k, v = F._in_projection_packed(query, key, value, in_proj_weight, in_proj_bias)
    is_decoder = query is not key

    # prep attention mask, ensure attn_mask's dim is 3
    if attn_mask is not None and attn_mask.dim() == 2:
        attn_mask = attn_mask.unsqueeze(0)

    # merge key padding and attention masks
    if key_padding_mask is not None:
        key_padding_mask = key_padding_mask.view(bsz, 1, 1, -1).expand(-1, num_heads, -1, -1).reshape(bsz * num_heads, 1, -1)

        if attn_mask is None:
            attn_mask = key_padding_mask
        elif attn_mask.dtype == torch.bool:
            attn_mask = attn_mask.logical_or(key_padding_mask)
        else:
            attn_mask = attn_mask.masked_fill(key_padding_mask, float("-inf"))

    # convert mask to float
    if attn_mask is not None and attn_mask.dtype == torch.bool:
        new_attn_mask = torch.zeros_like(attn_mask, dtype=q.dtype)
        new_attn_mask.masked_fill_(attn_mask, float("-inf"))
        attn_mask = new_attn_mask

    # (deep breath) calculate attention and out projection
    # q: (bsz * head_num, tgt_len, head_dim)
    # k_cache: (bsz, head_num, src_len, head_dim)
    # v_cache: (bsz, head_num, src_len, head_dim)
    q = q.contiguous().view(tgt_len, bsz * num_heads, head_dim).transpose(0, 1)  # (bsz * num_heads, tgt_len, head_dim)
    k = k.contiguous().view(k.shape[0], bsz * num_heads, head_dim).transpose(0, 1)
    v = v.contiguous().view(v.shape[0], bsz * num_heads, head_dim).transpose(0, 1)

    _, _, E = q.shape
    q_scaled = q / math.sqrt(E)

    if is_decoder:
        k = k_cache.view(bsz * num_heads, src_len, head_dim)
        v = v_cache.view(bsz * num_heads, src_len, head_dim)
    else:
        k = torch.cat([k_cache.reshape(bsz * num_heads, src_len, head_dim), k], dim=1)
        v = torch.cat([v_cache.reshape(bsz * num_heads, src_len, head_dim), v], dim=1)

    if attn_mask is not None:
        attn_output_weights = torch.baddbmm(attn_mask, q_scaled, k.transpose(-2, -1))
    else:
        attn_output_weights = torch.bmm(q_scaled, k.transpose(-2, -1))
    attn_output_weights = F.softmax(attn_output_weights, dim=-1)
    if training:
        attn_output_weights = F.dropout(attn_output_weights, p=dropout_p)

    attn_output = torch.bmm(attn_output_weights, v)
    attn_output = attn_output.transpose(0, 1).contiguous().view(tgt_len * bsz, embed_dim)
    attn_output = F.linear(attn_output, out_proj_weight, out_proj_bias)
    attn_output = attn_output.view(tgt_len, bsz, attn_output.size(1))

    new_k_cache = k.reshape(bsz, num_heads, -1, head_dim)
    new_v_cache = v.reshape(bsz, num_heads, -1, head_dim)
    return attn_output, (new_k_cache, new_v_cache)


def forward_attention(
    query,
    key,
    value,
    embed_dim_to_check,
    num_heads,
    in_proj_weight,
    in_proj_bias,
    bias_k,
    bias_v,
    add_zero_attn,
    dropout_p,
    out_proj_weight,
    out_proj_bias,
    training: bool = True,
    key_padding_mask = None,
    need_weights = True,
    attn_mask = None,
    use_separate_proj_weight: bool = False,
    q_proj_weight = None,
    k_proj_weight = None,
    v_proj_weight = None,
    static_k = None,
    static_v = None,
    average_attn_weights: bool = True,
):
    is_batched = F._mha_shape_check(query, key, value, key_padding_mask, attn_mask, num_heads)

    # For unbatched input, we unsqueeze at the expected batch-dim to pretend that the input
    # is batched, run the computation and before returning squeeze the
    # batch dimension so that the output doesn't carry this temporary batch dimension.
    if not is_batched:
        # unsqueeze if the input is unbatched
        query = query.unsqueeze(1)
        key = key.unsqueeze(1)
        value = value.unsqueeze(1)
        if key_padding_mask is not None:
            key_padding_mask = key_padding_mask.unsqueeze(0)

    # set up shape vars
    tgt_len, bsz, embed_dim = query.shape
    src_len, _, _ = key.shape
    assert embed_dim == embed_dim_to_check, \
        f"was expecting embedding dimension of {embed_dim_to_check}, but got {embed_dim}"
    if isinstance(embed_dim, torch.Tensor):
        # embed_dim can be a tensor when JIT tracing
        head_dim = embed_dim.div(num_heads, rounding_mode='trunc')
    else:
        head_dim = embed_dim // num_heads
    assert head_dim * num_heads == embed_dim, f"embed_dim {embed_dim} not divisible by num_heads {num_heads}"
    if use_separate_proj_weight:
        # allow MHA to have different embedding dimensions when separate projection weights are used
        assert key.shape[:2] == value.shape[:2], \
            f"key's sequence and batch dims {key.shape[:2]} do not match value's {value.shape[:2]}"
    else:
        assert key.shape == value.shape, f"key shape {key.shape} does not match value shape {value.shape}"

    #
    # compute in-projection
    #
    if not use_separate_proj_weight:
        assert in_proj_weight is not None, "use_separate_proj_weight is False but in_proj_weight is None"
        q, k, v = F._in_projection_packed(query, key, value, in_proj_weight, in_proj_bias)
    else:
        assert q_proj_weight is not None, "use_separate_proj_weight is True but q_proj_weight is None"
        assert k_proj_weight is not None, "use_separate_proj_weight is True but k_proj_weight is None"
        assert v_proj_weight is not None, "use_separate_proj_weight is True but v_proj_weight is None"
        if in_proj_bias is None:
            b_q = b_k = b_v = None
        else:
            b_q, b_k, b_v = in_proj_bias.chunk(3)
        q, k, v = F._in_projection(query, key, value, q_proj_weight, k_proj_weight, v_proj_weight, b_q, b_k, b_v)

    # prep attention mask
    if attn_mask is not None:
        if attn_mask.dtype == torch.uint8:
            F.warnings.warn("Byte tensor for attn_mask in nn.MultiheadAttention is deprecated. Use bool tensor instead.")
            attn_mask = attn_mask.to(torch.bool)
        else:
            assert attn_mask.is_floating_point() or attn_mask.dtype == torch.bool, \
                f"Only float, byte, and bool types are supported for attn_mask, not {attn_mask.dtype}"
        # ensure attn_mask's dim is 3
        if attn_mask.dim() == 2:
            correct_2d_size = (tgt_len, src_len)
            if attn_mask.shape != correct_2d_size:
                raise RuntimeError(f"The shape of the 2D attn_mask is {attn_mask.shape}, but should be {correct_2d_size}.")
            attn_mask = attn_mask.unsqueeze(0)
        elif attn_mask.dim() == 3:
            correct_3d_size = (bsz * num_heads, tgt_len, src_len)
            if attn_mask.shape != correct_3d_size:
                raise RuntimeError(f"The shape of the 3D attn_mask is {attn_mask.shape}, but should be {correct_3d_size}.")
        else:
            raise RuntimeError(f"attn_mask's dimension {attn_mask.dim()} is not supported")

    # add bias along batch dimension (currently second)
    if bias_k is not None and bias_v is not None:
        assert static_k is None, "bias cannot be added to static key."
        assert static_v is None, "bias cannot be added to static value."
        k = torch.cat([k, bias_k.repeat(1, bsz, 1)])
        v = torch.cat([v, bias_v.repeat(1, bsz, 1)])
        if attn_mask is not None:
            attn_mask = F.pad(attn_mask, (0, 1))
        if key_padding_mask is not None:
            key_padding_mask = F.pad(key_padding_mask, (0, 1))
    else:
        assert bias_k is None
        assert bias_v is None

    #
    # reshape q, k, v for multihead attention and make em batch first
    #
    q = q.contiguous().view(tgt_len, bsz * num_heads, head_dim).transpose(0, 1)
    if static_k is None:
        k = k.contiguous().view(k.shape[0], bsz * num_heads, head_dim).transpose(0, 1)
    else:
        # TODO finish disentangling control flow so we don't do in-projections when statics are passed
        assert static_k.size(0) == bsz * num_heads, \
            f"expecting static_k.size(0) of {bsz * num_heads}, but got {static_k.size(0)}"
        assert static_k.size(2) == head_dim, \
            f"expecting static_k.size(2) of {head_dim}, but got {static_k.size(2)}"
        k = static_k
    if static_v is None:
        v = v.contiguous().view(v.shape[0], bsz * num_heads, head_dim).transpose(0, 1)
    else:
        # TODO finish disentangling control flow so we don't do in-projections when statics are passed
        assert static_v.size(0) == bsz * num_heads, \
            f"expecting static_v.size(0) of {bsz * num_heads}, but got {static_v.size(0)}"
        assert static_v.size(2) == head_dim, \
            f"expecting static_v.size(2) of {head_dim}, but got {static_v.size(2)}"
        v = static_v

    # add zero attention along batch dimension (now first)
    if add_zero_attn:
        zero_attn_shape = (bsz * num_heads, 1, head_dim)
        k = torch.cat([k, torch.zeros(zero_attn_shape, dtype=k.dtype, device=k.device)], dim=1)
        v = torch.cat([v, torch.zeros(zero_attn_shape, dtype=v.dtype, device=v.device)], dim=1)
        if attn_mask is not None:
            attn_mask = F.pad(attn_mask, (0, 1))
        if key_padding_mask is not None:
            key_padding_mask = F.pad(key_padding_mask, (0, 1))

    # update source sequence length after adjustments
    src_len = k.size(1)

    # merge key padding and attention masks
    if key_padding_mask is not None:
        assert key_padding_mask.shape == (bsz, src_len), \
            f"expecting key_padding_mask shape of {(bsz, src_len)}, but got {key_padding_mask.shape}"
        key_padding_mask = key_padding_mask.view(bsz, 1, 1, src_len).   \
            expand(-1, num_heads, -1, -1).reshape(bsz * num_heads, 1, src_len)
        if attn_mask is None:
            attn_mask = key_padding_mask
        elif attn_mask.dtype == torch.bool:
            attn_mask = attn_mask.logical_or(key_padding_mask)
        else:
            attn_mask = attn_mask.masked_fill(key_padding_mask, float("-inf"))

    # convert mask to float
    if attn_mask is not None and attn_mask.dtype == torch.bool:
        new_attn_mask = torch.zeros_like(attn_mask, dtype=q.dtype)
        new_attn_mask.masked_fill_(attn_mask, float("-inf"))
        attn_mask = new_attn_mask

    # adjust dropout probability
    if not training:
        dropout_p = 0.0

    #
    # (deep breath) calculate attention and out projection
    #

    B, Nt, E = q.shape
    q_scaled = q / math.sqrt(E)
    if attn_mask is not None:
        attn_output_weights = torch.baddbmm(attn_mask, q_scaled, k.transpose(-2, -1))
    else:
        attn_output_weights = torch.bmm(q_scaled, k.transpose(-2, -1))
    attn_output_weights = F.softmax(attn_output_weights, dim=-1)
    if dropout_p > 0.0:
        attn_output_weights = F.dropout(attn_output_weights, p=dropout_p)

    attn_output = torch.bmm(attn_output_weights, v)

    attn_output = attn_output.transpose(0, 1).contiguous().view(tgt_len * bsz, embed_dim)
    attn_output = F.linear(attn_output, out_proj_weight, out_proj_bias)
    attn_output = attn_output.view(tgt_len, bsz, attn_output.size(1))

    if not is_batched:
        # squeeze the output if input was unbatched
        attn_output = attn_output.squeeze(1)

    k_cache = k.view(bsz, num_heads, src_len, head_dim)
    v_cache = v.view(bsz, num_heads, src_len, head_dim)
    return attn_output, (k_cache, v_cache)
