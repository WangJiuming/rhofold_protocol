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
import time
import logging

from rhofold.model.embedders import MSAEmbedder, RecyclingEmbedder
from rhofold.model.e2eformer import E2EformerStack
from rhofold.model.structure_module import StructureModule
from rhofold.model.heads import DistHead, SSHead, pLDDTHead
from rhofold.utils.tensor_utils import add
from rhofold.utils import exists
from rhofold.model.primitives import get_and_reset_sdpa_stats


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


    def forward_cords(self, tokens, single_fea, pair_fea, seq, *, profile: bool = False, logger: logging.Logger = None):

        output = self.structure_module.forward(seq, tokens, { "single": single_fea, "pair": pair_fea }, profile=profile, logger=logger)
        output['plddt'] = self.plddt_head(output['single'][-1])

        return output

    def forward_heads(self, pair_fea):

        output = {}
        output['ss'] = self.ss_head(pair_fea.float())
        output['p'], output['c4_'], output['n'] = self.dist_head(pair_fea.float())

        return output

    def forward_one_cycle(self, tokens, rna_fm_tokens, recycling_inputs, seq, *, profile: bool = False, logger: logging.Logger = None):
        '''
        Args:
            tokens: [bs, seq_len, c_z]
            rna_fm_tokens: [bs, seq_len, c_z]
        '''

        device = tokens.device

        msa_tokens_pert = tokens[:, :self.config.globals.msa_depth]

        # Optional detailed timing per recycle
        timings = {}
        def _sync():
            # Ensure accurate timings on async backends; ignore if not available
            try:
                if tokens.is_cuda:
                    torch.cuda.synchronize()
                # torch.mps.synchronize is available on recent PyTorch; guard it
                if hasattr(torch, 'mps') and hasattr(torch.mps, 'synchronize'):
                    torch.mps.synchronize()
            except Exception:
                pass

        if profile:
            _sync(); t0 = time.time()
        msa_fea, pair_fea = self.msa_embedder.forward(tokens=msa_tokens_pert,
                                                      rna_fm_tokens=rna_fm_tokens,
                                                      is_BKL=True)
        if profile:
            _sync(); timings['embed_msa'] = time.time() - t0

        if exists(self.recycle_embnet) and exists(recycling_inputs):
            if profile:
                _sync(); t1 = time.time()
            msa_fea_up, pair_fea_up = self.recycle_embnet(recycling_inputs['single_fea'],
                                                          recycling_inputs['pair_fea'],
                                                          recycling_inputs["cords_c1'"])
            msa_fea[..., 0, :, :] += msa_fea_up
            pair_fea = add(pair_fea, pair_fea_up, inplace=False)
            if profile:
                _sync(); timings['recycle_embed'] = time.time() - t1

        if profile:
            _sync(); t2 = time.time()
        msa_fea, pair_fea, single_fea = self.e2eformer(
            m=msa_fea,
            z=pair_fea,
            msa_mask=torch.ones(msa_fea.shape[:3]).to(device),
            pair_mask=torch.ones(pair_fea.shape[:3]).to(device),
            chunk_size=None,
            profile=profile,
            logger=logger,
        )
        if profile:
            _sync(); timings['e2eformer'] = time.time() - t2
            # Log sub-totals if available
            if logger is not None and hasattr(self.e2eformer, 'last_profile_totals'):
                subtotals = self.e2eformer.last_profile_totals
                logger.info("  E2Eformer sub: "
                            f"msa_att_row={subtotals.get('msa_att_row', 0.0):.3f}s, "
                            f"msa_att_col={subtotals.get('msa_att_col', 0.0):.3f}s, "
                            f"opm={subtotals.get('opm', 0.0):.3f}s, "
                            f"tri_mul_out={subtotals.get('tri_mul_out', 0.0):.3f}s, "
                            f"tri_mul_in={subtotals.get('tri_mul_in', 0.0):.3f}s, "
                            f"tri_att_start={subtotals.get('tri_att_start', 0.0):.3f}s, "
                            f"tri_att_end={subtotals.get('tri_att_end', 0.0):.3f}s, "
                            f"pair_transition={subtotals.get('pair_transition', 0.0):.3f}s")
                # SDPA usage summary
                sdpa_stats = get_and_reset_sdpa_stats()
                if sdpa_stats:
                    parts = []
                    for tag in sorted(sdpa_stats.keys()):
                        st = sdpa_stats[tag]
                        parts.append(f"{tag}: used={st['used']}, fallback={st['fallback']}")
                    logger.info("  SDPA usage: " + "; ".join(parts))

        if profile:
            _sync(); t3 = time.time()
        output = self.forward_cords(tokens, single_fea, pair_fea, seq, profile=profile, logger=logger)
        if profile:
            _sync(); timings['structure'] = time.time() - t3
            if logger is not None and 'timings_structure_sub' in output:
                st = output['timings_structure_sub']
                logger.info("  Structure sub: "
                            f"ipa={st.get('ipa', 0.0):.3f}s, "
                            f"transition={st.get('transition', 0.0):.3f}s, "
                            f"bb_update={st.get('bb_update', 0.0):.3f}s, "
                            f"angle_resnet={st.get('angle_resnet', 0.0):.3f}s, "
                            f"converter_build={st.get('converter_build', 0.0):.3f}s, "
                            f"refinenet={st.get('refinenet', 0.0):.3f}s")

        if profile:
            _sync(); t4 = time.time()
        output.update(self.forward_heads(pair_fea))
        if profile:
            _sync(); timings['heads'] = time.time() - t4

        recycling_outputs = {
            'single_fea': msa_fea[..., 0, :, :].detach(),
            'pair_fea': pair_fea.detach(),
            "cords_c1'": output["cords_c1'"][-1].detach(),
        }
        if profile:
            # Provide a total for this recycle for convenience
            timings['recycle_total'] = sum(timings.values())
            output['timings'] = timings
            if logger is not None:
                logger.info(f"Recycle timings: "
                            f"embed_msa={timings.get('embed_msa', 0):.3f}s, "
                            f"recycle_embed={timings.get('recycle_embed', 0):.3f}s, "
                            f"e2eformer={timings.get('e2eformer', 0):.3f}s, "
                            f"structure={timings.get('structure', 0):.3f}s, "
                            f"heads={timings.get('heads', 0):.3f}s, "
                            f"total={timings['recycle_total']:.3f}s")

        return output, recycling_outputs

    def forward(self,
                tokens,
                rna_fm_tokens,
                seq,
                *,
                profile: bool = False,
                logger: logging.Logger = None,
                **kwargs):

        """Perform the forward pass.

        Args:

        Returns:
        """

        recycling_inputs = None

        outputs = []
        totals = {'embed_msa': 0.0, 'recycle_embed': 0.0, 'e2eformer': 0.0, 'structure': 0.0, 'heads': 0.0, 'recycle_total': 0.0}
        for _r in range(self.config.model.recycling_embedder.recycles):
            output, recycling_inputs = \
                self.forward_one_cycle(tokens, rna_fm_tokens, recycling_inputs, seq, profile=profile, logger=logger)
            outputs.append(output)
            if profile and 'timings' in output:
                for k in totals:
                    totals[k] += float(output['timings'].get(k, 0.0))

        if profile and logger is not None and self.config.model.recycling_embedder.recycles > 0:
            logger.info("Forward totals: "
                        f"embed_msa={totals['embed_msa']:.3f}s, "
                        f"recycle_embed={totals['recycle_embed']:.3f}s, "
                        f"e2eformer={totals['e2eformer']:.3f}s, "
                        f"structure={totals['structure']:.3f}s, "
                        f"heads={totals['heads']:.3f}s, "
                        f"total={totals['recycle_total']:.3f}s")

        return outputs
