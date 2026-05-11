# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import torch.nn as nn

from rhofold.model.embedders import MSAEmbedder, RecyclingEmbedder
from rhofold.model.e2eformer import E2EformerStack
from rhofold.model.structure_module import StructureModule
from rhofold.model.heads import DistHead, SSHead, pLDDTHead
from rhofold.utils.tensor_utils import add
from rhofold.utils import exists


class RhoFold(nn.Module):
    """The rhofold network"""

    def __init__(self, config):
        """Constructor function."""

        super().__init__()

        self.config = config

        self.msa_embedder = MSAEmbedder(
            **config.model.msa_embedder,
        )
        self.e2eformer = E2EformerStack(
            **config.model.e2eformer_stack,
        )
        self.structure_module = StructureModule(
            **config.model.structure_module,
        )
        self.recycle_embnet = RecyclingEmbedder(
            **config.model.recycling_embedder,
        )
        self.dist_head = DistHead(
            **config.model.heads.dist,
        )
        self.ss_head = SSHead(
            **config.model.heads.ss,
        )
        self.plddt_head = pLDDTHead(
            **config.model.heads.plddt,
        )


    def forward_cords(self, tokens, single_fea, pair_fea, seq, res_mask=None):

        output = self.structure_module.forward(
            seq,
            tokens,
            {"single": single_fea, "pair": pair_fea},
            mask=res_mask,
        )
        output['plddt'] = self.plddt_head(output['single'][-1])

        return output

    def forward_heads(self, pair_fea):

        output = {}
        output['ss'] = self.ss_head(pair_fea.float())
        output['p'], output['c4_'], output['n'] = self.dist_head(pair_fea.float())

        return output

    def forward_one_cycle(self, tokens, rna_fm_tokens, recycling_inputs, seq,
                          msa_mask=None, pair_mask=None, res_mask=None):
        '''
        Args:
            tokens: [B, K, L] MSA token ids (may include padding via padding_idx).
            rna_fm_tokens: [B, L] RNA-FM token ids (may include padding via padding_idx).
            seq: str or List[str] of length B. Each entry is the unpadded RNA sequence
                 (only A/U/G/C residues, no padding) for the corresponding batch element.
            msa_mask: [B, K, L] bool/float mask, 1 for real positions, 0 for padding.
                      If None, defaults to all ones (back-compat with single-seq inference).
            pair_mask: [B, L, L] mask on pair positions. Defaults to outer product of res_mask.
            res_mask: [B, L] mask on residue positions. Defaults to all ones.
        '''

        device = tokens.device

        msa_tokens_pert = tokens[:, :self.config.globals.msa_depth]
        if msa_mask is not None:
            msa_mask = msa_mask[:, :self.config.globals.msa_depth]

        msa_fea, pair_fea = self.msa_embedder.forward(tokens=msa_tokens_pert,
                                                      rna_fm_tokens=rna_fm_tokens,
                                                      is_BKL=True)

        # Build masks for E2EformerStack. Default to ones for back-compat with
        # the single-sequence inference path.
        if res_mask is None:
            res_mask = torch.ones(msa_fea.shape[0], msa_fea.shape[2],
                                  device=device, dtype=msa_fea.dtype)
        if msa_mask is None:
            msa_mask = torch.ones(msa_fea.shape[:3], device=device, dtype=msa_fea.dtype)
        else:
            msa_mask = msa_mask.to(device=device, dtype=msa_fea.dtype)
        if pair_mask is None:
            pair_mask = res_mask[:, :, None] * res_mask[:, None, :]
        pair_mask = pair_mask.to(device=device, dtype=msa_fea.dtype)
        res_mask = res_mask.to(device=device, dtype=msa_fea.dtype)

        if exists(self.recycle_embnet) and exists(recycling_inputs):
            msa_fea_up, pair_fea_up = self.recycle_embnet(recycling_inputs['single_fea'],
                                                          recycling_inputs['pair_fea'],
                                                          recycling_inputs["cords_c1'"])
            msa_fea[..., 0, :, :] += msa_fea_up
            pair_fea = add(pair_fea, pair_fea_up, inplace=False)

        msa_fea, pair_fea, single_fea = self.e2eformer(
            m=msa_fea,
            z=pair_fea,
            msa_mask=msa_mask,
            pair_mask=pair_mask,
            chunk_size=None,
        )

        output = self.forward_cords(tokens, single_fea, pair_fea, seq, res_mask=res_mask)

        output.update(self.forward_heads(pair_fea))

        recycling_outputs = {
            'single_fea': msa_fea[..., 0, :, :].detach(),
            'pair_fea': pair_fea.detach(),
            "cords_c1'": output["cords_c1'"][-1].detach(),
        }

        return output, recycling_outputs

    def forward(self,
                tokens,
                rna_fm_tokens,
                seq,
                msa_mask=None,
                pair_mask=None,
                res_mask=None,
                **kwargs):

        """Perform the forward pass.

        Single-sequence usage (back-compat): pass tokens [1, K, L], rna_fm_tokens [1, L],
        seq as a string. Masks are all-ones by default.

        Batched usage: pass tokens [B, max_K, max_L] padded with padding_idx,
        rna_fm_tokens [B, max_L] padded with padding_idx, seq as a List[str] of
        length B (each entry the unpadded sequence), and explicit masks.

        Args:

        Returns:
        """

        recycling_inputs = None

        outputs = []
        for _r in range(self.config.model.recycling_embedder.recycles):
            output, recycling_inputs = \
                self.forward_one_cycle(tokens, rna_fm_tokens, recycling_inputs, seq,
                                       msa_mask=msa_mask, pair_mask=pair_mask,
                                       res_mask=res_mask)
            outputs.append(output)

        return outputs
