#!/usr/bin/env python

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import torch as th
import torch.nn as nn

from parlai.core.utils import neginf
from parlai.agents.transformer.modules import TransformerGeneratorModel
import nestedtensor

th = nestedtensor.nested.monkey_patch(th)


def universal_sentence_embedding(sentences, dim, sqrt=True):
    """
    Perform Universal Sentence Encoder averaging (https://arxiv.org/abs/1803.11175).

     This is really just sum / sqrt(len).

     :param Tensor sentences: an N x T x D of Transformer outputs. Note this is
        the exact output of TransformerEncoder, but has the time axis first
    :param ByteTensor: an N x T binary matrix of paddings

    :return: an N x D matrix of sentence embeddings
    :rtype Tensor:
    """
    sentence_sums = sentences.sum(dim)
    divisor = th.nested_tensor(sentences.nested_size(dim)).to(th.float)
    if sqrt:
        divisor = divisor.sqrt()
    sentence_sums = sentence_sums.div(divisor)
    return sentence_sums


class EndToEndModel(TransformerGeneratorModel):
    def __init__(self, opt, dictionary):
        super().__init__(opt, dictionary)
        self.encoder = ContextKnowledgeEncoder(self.encoder)
        self.decoder = ContextKnowledgeDecoder(self.decoder)

    def reorder_encoder_states(self, encoder_out, indices):
        enc, mask, ckattn = encoder_out
        if not th.is_tensor(indices):
            indices = th.LongTensor(indices).to(enc.device)
        enc = th.index_select(enc, 0, indices)
        mask = th.index_select(mask, 0, indices)
        ckattn = th.index_select(ckattn, 0, indices)
        return enc, mask, ckattn


class ContextKnowledgeEncoder(nn.Module):
    def __init__(self, transformer):
        super().__init__()
        # The transformer takes care of most of the work, but other modules
        # expect us to have an embeddings available
        self.embeddings = transformer.embeddings
        self.embed_dim = transformer.embeddings.embedding_dim
        self.transformer = transformer

    def forward(self, src_tokens, know_tokens, ck_mask, cs_ids, use_cs_ids):
        # encode the context, pretty basic
        context_encoded, context_mask = self.transformer(src_tokens)
        nested_context_encoded = th.nested_tensor_from_tensor_mask(
            context_encoded, context_mask, nested_dim=1)

        N, K, Tk = know_tokens.size()
        know_flat = know_tokens.reshape(-1, Tk)
        know_encoded, know_mask = self.transformer(know_flat)
        know_encoded = know_encoded.reshape(N, K, Tk, -1)
        know_mask = know_mask.reshape(N, K, Tk)

        nested_know_encoded = th.nested_tensor_from_tensor_mask(
            know_encoded, know_mask, nested_dim=2)

        # compute our sentence embeddings for context and knowledge
        context_use = universal_sentence_embedding(nested_context_encoded, 1)
        know_use = universal_sentence_embedding(nested_know_encoded, 2)

        context_use = context_use.div(np.sqrt(self.embed_dim))
        know_use = know_use.div(np.sqrt(self.embed_dim))

        know_use = know_use.to_tensor(dim=1)
        nested_ck_attn = th.mv(know_use, context_use)

        if not use_cs_ids:
            # if we're not given the true chosen_sentence (test time), pick our
            # best guess
            cs_ids = nested_ck_attn.argmax(1).to_tensor()

        # pick the true chosen sentence.
        # NOTE: NestedTensor doesn't support advanced indexing for now.
        nested_cs_encoded = th.nested_tensor(
            [nested_know_encoded[i][cs_ids[i]] for i in range(len(cs_ids))])

        # Convert it back to tensors + masks for compatability
        cs_encoded, cs_mask = nested_cs_encoded.to_tensor_mask()
        context_encoded, context_mask = nested_context_encoded.to_tensor_mask(
            mask_dim=2)
        ck_attn, ck_attn_mask = nested_ck_attn.to_tensor_mask()

        # finally, concatenate it all
        full_enc = th.cat([cs_encoded, context_encoded], dim=1)
        full_mask = th.cat([cs_mask, context_mask], dim=1)
        print(1)

        # also return the knowledge selection mask for the loss
        return full_enc, full_mask, ck_attn


class ContextKnowledgeDecoder(nn.Module):
    def __init__(self, transformer):
        super().__init__()
        self.transformer = transformer

    def forward(self, input, encoder_state, incr_state=None):
        # our CK Encoder returns an extra output which the Transformer decoder
        # doesn't expect (the knowledge selection mask). Just chop it off
        encoder_output, encoder_mask, _ = encoder_state
        return self.transformer(input, (encoder_output, encoder_mask), incr_state)
