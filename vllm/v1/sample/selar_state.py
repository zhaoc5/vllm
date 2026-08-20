# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""SeLaR: entropy-gated latent reasoning.

SeLaR (the ``selar`` method) reads each step's uncertainty off the *sampling*
distribution -- after penalties, temperature and the top_k/top_p/min_p filters
-- as the entropy of the renormalised top-``selar_topk`` head, scaled into
``[0, 1]`` by its ceiling ``log k``. A step whose signal clears the threshold
feeds back a *latent* input instead of the sampled token's embedding: the
probability-weighted mixture of the head embeddings, displaced away from the
most probable token's embedding by a contrastive push that grows with the
signal. Sampling and the recorded trace stay ordinary throughout, and steps
that land on a math token stay discrete.

Unlike Soft Thinking and SwiReasoning there is no cross-step machine: every
step gates independently, so this holder only carries each row's config and
the one-step directive between the sampler and the model runner. The shared
contracts still apply: ``sync_batch`` follows the BatchUpdate order, and rows
whose sampled token the runner discards are skipped so a mid-prompt chunk
cannot leave a stale directive behind for the first real decode step.
"""

import math

import torch

from vllm.logger import init_logger
from vllm.v1.sample.logits_processor.interface import BatchUpdate, MoveDirectionality
from vllm.v1.sample.soft_thinking import _apply_filters
from vllm.v1.sample.soft_thinking_state import _min_p_from_metadata

logger = init_logger(__name__)


def maybe_create_selar_state_holder(max_num_seqs: int, device: torch.device):
    """Always available: selar needs no reasoning tokens, only embeddings."""
    return SelarStateHolder(max_num_seqs, device)


class SelarStateHolder:
    """Carries each selar row's config and its one-step latent directive."""

    def __init__(self, max_num_seqs: int, device: torch.device):
        self.device = device
        self.max_num_seqs = max_num_seqs
        self._state: dict[int, dict] = {}
        self._skip_rows: frozenset[int] = frozenset()

    def has_tracked_requests(self) -> bool:
        return bool(self._state)

    def set_rows_without_a_real_decode(self, rows) -> None:
        self._skip_rows = frozenset(rows)

    def sync_batch(self, batch_update: BatchUpdate | None) -> None:
        if not batch_update:
            return
        for index in batch_update.removed:
            self._state.pop(index, None)

        for index, params, _prompt_tok_ids, _output_tok_ids in batch_update.added:
            if getattr(params, "selar", False):
                self._state[index] = {
                    "topk": params.selar_topk,
                    "threshold": params.selar_entropy_threshold,
                    "weight": params.selar_contrastive_weight,
                    "math_ids": frozenset(params.selar_math_token_ids or ()),
                    # The gate compares against the entropy ceiling of a k-way
                    # categorical.
                    "ceiling": math.log(float(params.selar_topk)),
                    # -- one-step directive --
                    "head_ids": None,
                    "head_weights": None,
                    "push": None,
                }
            else:
                self._state.pop(index, None)

        for i1, i2, direction in batch_update.moved:
            if direction == MoveDirectionality.SWAP:
                s1 = self._state.pop(i1, None)
                s2 = self._state.pop(i2, None)
                if s1 is not None:
                    self._state[i2] = s1
                if s2 is not None:
                    self._state[i1] = s2
            else:
                s = self._state.pop(i1, None)
                if s is not None:
                    self._state[i2] = s

    def _active_rows(self) -> list[int]:
        return sorted(i for i in self._state if i not in self._skip_rows)

    def prepare(self, logits, sampling_metadata) -> None:
        """Gate every active row on this step's sampling distribution.

        Runs before the sampler's own pipeline, because ``apply_temperature``
        divides the logits in place: ``logits`` here are the post-penalty,
        pre-temperature logits Soft Thinking hooks. The filtered distribution
        is recomputed with the metadata's temperature/top_k/top_p/min_p,
        exactly as the reference applies ``apply_sampling_filter`` before both
        its gate and its sampling; sampling itself is untouched and draws
        from an equivalent distribution.

        Stores each gated row's directive (head ids, renormalised weights and
        the contrastive push scale). :meth:`finalize` then clears it wherever
        the sampled token turned out to be a math symbol.
        """
        rows = self._active_rows()
        if not rows:
            return
        idx = torch.tensor(rows, device=logits.device, dtype=torch.long)
        sub = logits[idx].to(torch.float32)

        temperature = sampling_metadata.temperature
        if temperature is not None:
            t = temperature.reshape(-1)[idx]
            t = torch.where(t < 1e-5, torch.ones_like(t), t)
            sub = sub / t.unsqueeze(-1)
        top_k = sampling_metadata.top_k
        top_p = sampling_metadata.top_p
        min_p = _min_p_from_metadata(sampling_metadata, logits.shape[0])
        sub = _apply_filters(
            sub,
            None if top_k is None else top_k.reshape(-1)[idx],
            None if top_p is None else top_p.reshape(-1)[idx],
            None if min_p is None else min_p.reshape(-1)[idx],
        )
        probs = torch.softmax(sub, dim=-1)

        max_k = max(self._state[r]["topk"] for r in rows)
        head_probs, head_ids = torch.topk(probs, k=min(max_k, probs.shape[-1]), dim=-1)

        for pos, r in enumerate(rows):
            st = self._state[r]
            k = st["topk"]
            hp = head_probs[pos, :k]
            hw = hp / (hp.sum() + 1e-10)
            entropy = -torch.sum(hw * torch.log(hw + 1e-10)).item()
            signal = min(max(entropy / st["ceiling"], 0.0), 1.0)
            if signal >= st["threshold"]:
                st["head_ids"] = head_ids[pos, :k]
                st["head_weights"] = hw
                st["push"] = st["weight"] * signal
            else:
                st["head_ids"] = None
                st["head_weights"] = None
                st["push"] = None

    def finalize(self, sampled) -> None:
        """Close the gate wherever the sampled token is a math symbol --
        notation drifts badly when consumed as a mixture, so those steps feed
        the ordinary embedding no matter how uncertain the model was."""
        rows = [r for r in self._active_rows()
                if self._state[r]["head_ids"] is not None
                and self._state[r]["math_ids"]]
        if not rows:
            return
        sampled_list = sampled.tolist()
        for r in rows:
            st = self._state[r]
            if int(sampled_list[r]) in st["math_ids"]:
                st["head_ids"] = None
                st["head_weights"] = None
                st["push"] = None

    def embed_directives(self):
        """The latent inputs to overlay next step.

        Returns ``(rows, head_ids, head_weights, pushes)`` padded to the
        widest k in the batch (weights are zero past each row's own k, so the
        padding contributes nothing), or None when no row gated.
        """
        rows = [r for r in sorted(self._state)
                if self._state[r]["head_ids"] is not None]
        if not rows:
            return None
        width = max(self._state[r]["head_ids"].numel() for r in rows)
        ids, weights, pushes = [], [], []
        for r in rows:
            st = self._state[r]
            k = st["head_ids"].numel()
            pad = width - k
            ids.append(torch.nn.functional.pad(st["head_ids"], (0, pad)))
            weights.append(torch.nn.functional.pad(st["head_weights"], (0, pad)))
            pushes.append(st["push"])
        device = ids[0].device
        return (
            torch.tensor(rows, device=device, dtype=torch.long),
            torch.stack(ids),
            torch.stack(weights),
            torch.tensor(pushes, device=device, dtype=torch.float32),
        )
