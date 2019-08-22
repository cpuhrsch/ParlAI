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
        nested_context_encoded = th.tensor_mask_to_nested_tensor(
            context_encoded, context_mask)

        N, K, Tk = know_tokens.size()
        know_flat = know_tokens.reshape(-1, Tk)
        know_encoded, know_mask = self.transformer(know_flat)
        know_encoded = know_encoded.reshape(N, K, Tk, -1)
        know_mask = know_mask.reshape(N, K, Tk)

        know_lengths = ck_mask.sum(1)
        # Convert into a NestedTensor TODO: move this into tensor_mask_to_nested_tensor
        nested_know_encoded = th.nested_tensor([th.tensor_mask_to_nested_tensor(
            know_encoded[i][:know_lengths[i]], know_mask[i][:know_lengths[i]]) for i in range(len(know_mask))])

        print(know_mask)
        print(know_encoded)
        print(nested_know_encoded.size())
        print(nested_know_encoded.size(1))

        # Perform Universal Sentence Encoder averaging (https://arxiv.org/abs/1803.11175).
        # and normalization by embed_dim
        def divisor(t):
            if isinstance(t[0], tuple):
                return tuple(map(divisor, t))
            else:
                return th.tensor(float(self.embed_dim) * float(t[0])).sqrt()

        # compute our sentence embeddings for context and knowledge
        context_use = nested_context_encoded.sum(1)
        know_use = nested_know_encoded.sum(2)

        context_use_divisor = th.nested_tensor(
            divisor(nested_context_encoded.nested_size()))
        know_use_divisor = th.nested_tensor(
            divisor(nested_know_encoded.nested_size()))

        divisor2 = th.nested_tensor(list(map(th.tensor, nested_know_encoded.size(1)))).to(
            th.float).sqrt()

        context_use_divisor = th.nested_tensor(list(map(th.tensor, nested_context_encoded.size(1)))).to(
            th.float).mul(th.nested_tensor([th.tensor(256)] * 64)).sqrt()

        import pdb
        pdb.set_trace()

        context_use = context_use.div(context_use_divisor)
        know_use = know_use.div(know_use_divisor)

        know_use = th.nested_tensor(
            list(map(lambda x: x.to_tensor(), know_use.unbind())))
        nested_ck_attn = th.mv(know_use, context_use)

        if not use_cs_ids:
            # if we're not given the true chosen_sentence (test time), pick our
            # best guess
            cs_ids = nested_ck_attn.argmax(1).to_tensor()

        nested_cs_encoded = th.nested_tensor([nested_know_encoded.unbind()[i].unbind()[
            cs_ids[i]] for i in range(len(cs_ids))])

        # Convert it back to tensors + masks for compatability
        cs_encoded, cs_mask = nested_cs_encoded.to_tensor_mask()
        context_encoded, context_mask = nested_context_encoded.to_tensor_mask()
        ck_attn, ck_attn_mask = nested_ck_attn.to_tensor_mask()

        # finally, concatenate it all
        full_enc = th.cat([cs_encoded, context_encoded], dim=1)
        full_mask = th.cat([cs_mask, context_mask], dim=1)

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
