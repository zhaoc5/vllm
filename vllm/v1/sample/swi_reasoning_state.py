# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""SwiReasoning: entropy-trend switching between soft and discrete thinking.

SwiReasoning (the ``swir`` method) watches the entropy of the model's *raw*
next-token distribution and switches each row between two feedback regimes:
*soft*, where the input fed back is the full-vocabulary probability-weighted
mixture of token embeddings, and *normal*, where it is the sampled token's
embedding. A falling entropy (below the reference set at the last switch)
flips soft -> normal; a rising entropy flips back once the row has stayed put
for ``swir_window`` steps. Sampling itself is ordinary throughout -- unlike
Soft Thinking, the recorded token is always the sampled one, and only the
*embedding* fed back changes.

Switch steps blend the mixture with an anchor embedding (``<think>`` when
entering soft, ``</think>`` when leaving, a line break on the very first
step), and an optional switch budget forces convergence: after
``swir_max_switch_count`` soft->normal switches the row is fed ``</think>``
token by token, after twice that many it is fed a termination phrase and then
cut off a fixed number of tokens later.

This holder owns the per-request state; the sampler feeds it the raw logits
(before penalties -- the reference reads the model's own distribution) and the
sampled ids, and the model runner reads back per-row *directives* describing
the embedding to feed next step. It mirrors ``SoftThinkingStateHolder``'s
contracts: ``sync_batch`` follows the BatchUpdate order, and rows whose
sampled token the runner discards (chunked or resumed prefills) are skipped
via ``set_rows_without_a_real_decode``.
"""

import torch

from vllm.logger import init_logger
from vllm.v1.sample.logits_processor.interface import BatchUpdate, MoveDirectionality

logger = init_logger(__name__)


def maybe_create_swi_reasoning_state_holder(
    reasoning_config,
    max_num_seqs: int,
    device: torch.device,
):
    """None when the model declares no thinking-block tokens: the switch
    anchors are ``<think>`` and ``</think>``, and without them the method
    cannot blend or force convergence."""
    if (
        reasoning_config is None
        or not reasoning_config.reasoning_end_token_ids
        or not reasoning_config.reasoning_start_token_ids
    ):
        return None
    return SwiReasoningStateHolder(reasoning_config, max_num_seqs, device)


class SwiReasoningStateHolder:
    """Tracks each swir row's mode machine and carries its next-input directive."""

    def __init__(self, reasoning_config, max_num_seqs: int, device: torch.device):
        self.think_start_id = reasoning_config.reasoning_start_token_ids[-1]
        self.think_end_id = reasoning_config.reasoning_end_token_ids[-1]
        self.device = device
        self.max_num_seqs = max_num_seqs
        # req_index -> per-request config and state; absent means "not swir".
        self._state: dict[int, dict] = {}
        # Rows with no real decode this step; refreshed by the runner.
        self._skip_rows: frozenset[int] = frozenset()

    def has_tracked_requests(self) -> bool:
        return bool(self._state)

    def set_rows_without_a_real_decode(self, rows) -> None:
        self._skip_rows = frozenset(rows)

    def sync_batch(self, batch_update: BatchUpdate | None) -> None:
        """Track add/remove/move of rows; the documented order is removed,
        added, moved."""
        if not batch_update:
            return
        for index in batch_update.removed:
            self._state.pop(index, None)

        for index, params, _prompt_tok_ids, output_tok_ids in batch_update.added:
            if getattr(params, "swir", False):
                out = output_tok_ids or ()
                self._state[index] = {
                    # -- static per-request config --
                    "alpha0": params.swir_alpha,
                    "beta0": params.swir_beta,
                    "window": params.swir_window,
                    "max_switch": params.swir_max_switch_count,
                    "conv_ids": list(params.swir_convergence_token_ids or ()),
                    "term_ids": list(params.swir_termination_token_ids or ()),
                    "term_max": params.swir_termination_max_tokens,
                    "math_ids": frozenset(params.swir_math_token_ids or ()),
                    "linebreak_id": params.swir_linebreak_token_id,
                    # None means "until max_model_len", which this holder cannot
                    # see; a huge stand-in keeps the blend ramps at their floor,
                    # i.e. permanently "early in the generation".
                    "max_tokens": params.max_tokens or (1 << 20),
                    "stop_ids": sorted(
                        set(getattr(params, "all_stop_token_ids", None)
                            or params.stop_token_ids or ())
                    ),
                    # -- dynamic state --
                    # A preemption-resumed row restarts its entropy machine (the
                    # reference has no notion of resuming), but a row whose trace
                    # already closed its thinking block must stay locked normal.
                    "step": len(out),
                    "mode": 0,  # 0 = soft, 1 = normal
                    "stay": 0,
                    "ref": None,
                    "locked": self.think_end_id in out,
                    "switch_count": 0,
                    "queue": [],
                    "injecting": False,
                    "budget": -1,
                    # -- per-step transients --
                    "probs": None,     # raw fp32 distribution, set by observe_logits
                    "entropy": None,
                    "directive": None,  # what the runner feeds next step
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

    def observe_logits(self, logits: torch.Tensor) -> None:
        """Capture the raw distribution before penalties touch the logits.

        The reference reads its entropy signal and builds its mixture from the
        model's own distribution -- softmax of the raw logits, before the
        presence penalty and temperature -- so this must run at the very top
        of the sampler, before ``apply_logits_processors`` mutates the tensor
        in place.
        """
        rows = self._active_rows()
        if not rows:
            return
        idx = torch.tensor(rows, device=logits.device, dtype=torch.long)
        probs = torch.softmax(logits[idx].to(torch.float32), dim=-1)
        entropy = -(probs * torch.log(probs.clamp_min(1e-12))).sum(dim=-1)
        ent = entropy.tolist()
        for pos, r in enumerate(rows):
            self._state[r]["probs"] = probs[pos]
            self._state[r]["entropy"] = ent[pos]

    def step(self, sampled: torch.Tensor) -> torch.Tensor | None:
        """Advance every active row by one step; return forced token ids.

        Follows the reference's per-step order: the sampled token locks the
        row once it is ``</think>``; a pending injection queue overrides the
        token; the entropy machine then updates the mode; injection triggers
        set up *future* steps; finally the next-input directive is derived
        from the post-update mode.

        Returns ``[num_reqs]`` of ids to force (``-1`` where none), or None
        when no row is active.
        """
        # `entropy` marks "observed this step": `probs` outlives the step for
        # soft rows (the runner reads it next step), so it cannot be the marker.
        rows = [r for r in self._active_rows()
                if self._state[r]["entropy"] is not None]
        if not rows:
            return None
        forced = torch.full(
            (sampled.shape[0],), -1, device=sampled.device, dtype=torch.long
        )
        sampled_list = sampled.tolist()

        for r in rows:
            st = self._state[r]
            tok = int(sampled_list[r])
            if tok == self.think_end_id:
                st["locked"] = True

            # Injection planned on an earlier step overrides the sampled token.
            force_id = -1
            if st["injecting"] and st["queue"]:
                force_id = st["queue"].pop(0)
                if not st["queue"]:
                    st["injecting"] = False
                tok = force_id

            # The termination budget counts every step once armed; at zero the
            # row is cut off by forcing a stop token (the reference marks it
            # finished without emitting one -- the stop id detokenizes to
            # nothing, so the text is identical).
            if st["budget"] >= 0:
                st["budget"] -= 1
                if st["budget"] == 0 and st["stop_ids"]:
                    force_id = st["stop_ids"][0]

            # -- entropy trend machine --
            ent = st["entropy"]
            to_normal = to_soft = False
            if st["step"] == 0 or st["ref"] is None:
                st["ref"] = ent
            else:
                st["stay"] += 1
                allow_switch = st["stay"] >= st["window"]
                to_normal = st["mode"] == 0 and ent < st["ref"]
                to_soft = (st["mode"] == 1 and ent > st["ref"]
                           and allow_switch and not st["locked"])
                if to_normal:
                    st["mode"], st["stay"], st["ref"] = 1, 0, ent
                    st["switch_count"] += 1
                elif to_soft:
                    st["mode"], st["stay"], st["ref"] = 0, 0, ent

            # Convergence/termination triggers arm the queue for later steps.
            ms = st["max_switch"]
            if ms is not None and to_normal and st["step"] > 0:
                if ms <= st["switch_count"] <= 2 * ms and st["conv_ids"]:
                    st["queue"] = list(st["conv_ids"])
                    st["injecting"] = True
                elif st["switch_count"] > 2 * ms and st["term_ids"]:
                    st["queue"] = list(st["term_ids"])
                    st["injecting"] = True
                    # The reference sets the budget and decrements it once in
                    # the same step; arming it pre-decremented matches that.
                    st["budget"] = st["term_max"] - 1

            # -- next-input directive, from the post-update mode --
            is_normal = st["mode"] == 1 or st["locked"] or tok in st["math_ids"]
            ratio = float(st["step"]) / float(max(st["max_tokens"], 1))
            if to_normal:
                # Leaving soft: what goes back in is the mixture eased toward
                # </think>, not the sampled token's embedding.
                beta = st["beta0"] + (1.0 - st["beta0"]) * ratio
                st["directive"] = {"blend_id": self.think_end_id, "blend_w": beta}
            elif not is_normal:
                if st["step"] == 0:
                    st["directive"] = {"blend_id": st["linebreak_id"],
                                       "blend_w": 0.9}
                elif to_soft:
                    alpha = st["alpha0"] + (1.0 - st["alpha0"]) * ratio
                    st["directive"] = {"blend_id": self.think_start_id,
                                       "blend_w": alpha}
                else:
                    st["directive"] = {"blend_id": None, "blend_w": 1.0}
            else:
                # Plain discrete feedback: the ordinary token-id path embeds it.
                st["directive"] = None
                st["probs"] = None

            st["step"] += 1
            st["entropy"] = None
            if force_id >= 0:
                forced[r] = force_id
        return forced

    def embed_directives(self):
        """What the runner must overlay next step.

        Returns ``(rows, probs, blend_ids, blend_ws)`` -- rows into the
        sampling batch, the fp32 full-vocabulary weights of each row's
        mixture, and per-row blend anchors (-1 for none) with their weights:
        ``final = w * mixture + (1 - w) * E[anchor]``. None when no row feeds
        a mixture.
        """
        rows = [r for r in sorted(self._state)
                if self._state[r]["directive"] is not None
                and self._state[r]["probs"] is not None]
        if not rows:
            return None
        probs = torch.stack([self._state[r]["probs"] for r in rows])
        blend_ids = torch.tensor(
            [self._state[r]["directive"]["blend_id"]
             if self._state[r]["directive"]["blend_id"] is not None else -1
             for r in rows],
            device=probs.device, dtype=torch.long,
        )
        blend_ws = torch.tensor(
            [self._state[r]["directive"]["blend_w"] for r in rows],
            device=probs.device, dtype=torch.float32,
        )
        idx = torch.tensor(rows, device=probs.device, dtype=torch.long)
        return idx, probs, blend_ids, blend_ws
