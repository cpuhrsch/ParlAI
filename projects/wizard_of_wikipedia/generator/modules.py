#!/usr/bin/env python

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import torch as th
import torch.nn as nn
import math

from parlai.core.utils import neginf
from parlai.agents.transformer.modules import TransformerGeneratorModel

th = th.nested.monkey_patch(th)


def _universal_nested_sentence_embedding(nested_tensor, sqrt=True):
    """
    Perform Universal Sentence Encoder averaging (https://arxiv.org/abs/1803.11175).

    This is really just sum / sqrt(len).

    :param Tensor sentences: an N x T x D of Transformer outputs. Note this is
        the exact output of TransformerEncoder, but has the time axis first
    :param ByteTensor: an N x T binary matrix of paddings

    :return: an N x D matrix of sentence embeddings
    :rtype Tensor:
    """
    # need to mask out the padded chars
    sentence_sums = nested_tensor.sum(1)
    # TODO: How to turn this into an operation on top of nested_size
    divisor = list(map(lambda x: x[0], nested_tensor.nested_size()))
    if sqrt:
        divisor = list(map(lambda x: math.sqrt(x), divisor))
    divisor = th.nested_tensor(list(map(lambda x: th.tensor(x), divisor)))
    return sentence_sums.div(divisor)


class EndToEndModel(TransformerGeneratorModel):
    def __init__(self, opt, dictionary):
        super().__init__(opt, dictionary)
        self.encoder = _ContextKnowledgeEncoder(self.encoder)
        self.decoder = _ContextKnowledgeDecoder(self.decoder)

    def reorder_encoder_states(self, encoder_out, indices):
        enc, mask, ckattn = encoder_out
        if not th.is_tensor(indices):
            indices = th.LongTensor(indices).to(enc.device)
        enc = th.index_select(enc, 0, indices)
        mask = th.index_select(mask, 0, indices)
        ckattn = th.index_select(ckattn, 0, indices)
        return enc, mask, ckattn


class _ContextKnowledgeEncoder(nn.Module):
    def __init__(self, transformer):
        super().__init__()
        # The transformer takes care of most of the work, but other modules
        # expect us to have an embeddings available
        self.embeddings = transformer.embeddings
        self.embed_dim = transformer.embeddings.embedding_dim
        self.transformer = transformer

    def forward(self, src_tokens, know_tokens, ck_mask, cs_ids, use_cs_ids):
        # encode the context, pretty basic
        _context_encoded, _context_mask = self.transformer(src_tokens)
        _nested_context_encoded = th.tensor_mask_to_nested_tensor(
            _context_encoded, _context_mask)

        # make all the knowledge into a 2D matrix to encode
        N, K, Tk = know_tokens.size()
        know_flat = know_tokens.reshape(-1, Tk)
        _know_encoded, _know_mask = self.transformer(know_flat)
        _know_encoded = _know_encoded.reshape(N, K, Tk, -1)
        _know_mask = _know_mask.reshape(N, K, Tk)
        _nested_know_encoded = th.nested_tensor([th.tensor_mask_to_nested_tensor(
            _know_encoded[i], _know_mask[i]) for i in range(len(_know_mask))])

        # compute our sentence embeddings for context and knowledge

        context_use = _nested_context_encoded.sum(1)
        know_use = _nested_know_encoded.sum(2)
        import pdb
        pdb.set_trace()

        context_use = _universal_nested_sentence_embedding(
            _nested_context_encoded)
        # context_use = th.stack(context_use.unbind())

        know_use = _universal_nested_sentence_embedding(_nested_know_encoded)
        # know_use = th.stack(know_use.unbind())

        sqrt_embed_dim = math.sqrt(float(self.embed_dim))
        context_use = context_use.div(th.nested_tensor(
            [th.tensor(sqrt_embed_dim) for i in range(len(context_use))]))
        know_use = know_use.div(th.nested_tensor(
            [th.tensor(sqrt_embed_dim) for i in range(len(know_use))]))

        # # remash it back into the shape we need
        # know_use = know_use.reshape(N, know_tokens.size(1), self.embed_dim)

        ck_attn = th.bmm(know_use, context_use.unsqueeze(-1)).squeeze(-1)
        # fill with near -inf
        ck_attn.masked_fill_(~ck_mask, neginf(context_encoded.dtype))

        if not use_cs_ids:
            # if we're not given the true chosen_sentence (test time), pick our
            # best guess
            _, cs_ids = ck_attn.max(1)

        # pick the true chosen sentence. remember that TransformerEncoder outputs
        #   (batch, time, embed)
        # but because know_encoded is a flattened, it's really
        #   (N * K, T, D)
        # We need to compute the offsets of the chosen_sentences
        cs_offsets = th.arange(N, device=cs_ids.device) * K + cs_ids
        cs_encoded = know_encoded[cs_offsets]
        # but padding is (N * K, T)
        cs_mask = know_mask[cs_offsets]

        # finally, concatenate it all
        full_enc = th.cat([cs_encoded, context_encoded], dim=1)
        full_mask = th.cat([cs_mask, context_mask], dim=1)

        # also return the knowledge selection mask for the loss
        return full_enc, full_mask, ck_attn


class _ContextKnowledgeDecoder(nn.Module):
    def __init__(self, transformer):
        super().__init__()
        self.transformer = transformer

    def forward(self, input, encoder_state, incr_state=None):
        # our CK Encoder returns an extra output which the Transformer decoder
        # doesn't expect (the knowledge selection mask). Just chop it off
        encoder_output, encoder_mask, _ = encoder_state
        return self.transformer(input, (encoder_output, encoder_mask), incr_state)
