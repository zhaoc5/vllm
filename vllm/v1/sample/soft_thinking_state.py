# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Per-batch Soft Thinking state: which rows are still reasoning, and their
concept tokens.

The mixture a step produces is fed back on the *next* step, so this holder keeps
the last step's top-k for every row still inside its thinking block. The model
runner reads it when building ``inputs_embeds``; the sampler updates it.

Shaped after ``ThinkingBudgetStateHolder``, which solves the same bookkeeping
problem for thinking-token budgets and already resolves ``</think>`` from the
reasoning config.
"""

from typing import TYPE_CHECKING

import torch

from vllm.logger import init_logger
from vllm.v1.sample.logits_processor.interface import BatchUpdate, MoveDirectionality
from vllm.v1.sample.soft_thinking import ConceptTokens, update_cold_stop

if TYPE_CHECKING:
    from vllm.config.reasoning import ReasoningConfig

logger = init_logger(__name__)


def maybe_create_soft_thinking_state_holder(
    reasoning_config: "ReasoningConfig | None",
    max_num_seqs: int,
    device: torch.device,
) -> "SoftThinkingStateHolder | None":
    """None when the model declares no reasoning tokens, which makes the feature
    unusable: there is no ``</think>`` to end the thinking block with."""
    if reasoning_config is None or not reasoning_config.reasoning_end_token_ids:
        return None
    return SoftThinkingStateHolder(reasoning_config, max_num_seqs, device)


def _min_p_from_metadata(sampling_metadata, num_reqs: int) -> torch.Tensor | None:
    """The batch's per-request min_p, or None when no request asks for it.

    The processor resizes its tensor as requests come and go, so a length that
    does not match the batch is treated as absent rather than indexed into.
    """
    from vllm.v1.sample.logits_processor.builtin import MinPLogitsProcessor

    logitsprocs = getattr(sampling_metadata, "logitsprocs", None)
    if logitsprocs is None:
        return None
    for proc in logitsprocs.argmax_invariant:
        if isinstance(proc, MinPLogitsProcessor) and proc.min_p_count > 0:
            # Stored as [num_reqs, 1], ready to broadcast against logits; the
            # caller wants one value per request.
            min_p = proc.min_p.reshape(-1)
            if min_p.numel() == num_reqs:
                return min_p
    return None


class SoftThinkingStateHolder:
    """Tracks the thinking phase and carries concept tokens between steps."""

    def __init__(
        self,
        reasoning_config: "ReasoningConfig",
        max_num_seqs: int,
        device: torch.device,
    ):
        end_ids = reasoning_config.reasoning_end_token_ids
        # A multi-token </think> cannot be forced in one step, and Cold Stop has
        # nothing else to emit; take the final id, which is the one that closes.
        self.think_end_token_id = end_ids[-1]
        self.device = device
        self.max_num_seqs = max_num_seqs

        # req_index -> per-request config and state. Absent means "not soft".
        # The concept token lives here too, so sync_batch's move handling carries
        # it with the row; keeping it in a separate tensor keyed by the previous
        # step's indices would go stale the moment the batch is compacted.
        self._state: dict[int, dict] = {}
        # Rows whose sampled token the runner will discard this step; refreshed
        # by the model runner every step, before the sampler runs.
        self._skip_rows: frozenset[int] = frozenset()

    def has_tracked_requests(self) -> bool:
        return bool(self._state)

    def sync_batch(self, batch_update: BatchUpdate | None) -> None:
        """Track add/remove/move of rows, mirroring the logits-processor contract."""
        if not batch_update:
            return
        for index in batch_update.removed:
            self._state.pop(index, None)

        for index, params, _prompt_tok_ids, output_tok_ids in batch_update.added:
            if getattr(params, "soft_thinking", False):
                self._state[index] = {
                    "soft_topk": params.soft_topk,
                    "entropy_threshold": params.soft_entropy_threshold,
                    "patience": params.soft_patience,
                    # Qwen3.5's chat template pre-fills "<think>\n" into the
                    # prompt, so generation starts inside the thinking block. A
                    # preempted request comes back through here with its outputs
                    # so far; one that already closed its thinking block must
                    # not re-enter it -- concept-mixing the answer phase is
                    # exactly what Soft Thinking says not to do. A row resumed
                    # *mid*-thinking restarts with a fresh Cold Stop run and,
                    # for its first step, the recorded argmax instead of the
                    # lost mixture -- the re-prefill already replayed the
                    # discrete trace, so that is the consistent continuation.
                    "in_thinking": self.think_end_token_id
                    not in (output_tok_ids or ()),
                    "low_entropy_steps": 0,
                    "stop_ids": set(getattr(params, "all_stop_token_ids", None)
                                    or params.stop_token_ids or ()),
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

    def set_rows_without_a_real_decode(self, rows) -> None:
        """Mark the rows whose sampled token the runner will discard this step.

        A chunked first prefill, a re-prefill after preemption, and a row
        scheduled zero tokens all still occupy a sampler row, but their logits
        come from a mid-prompt position and the runner throws the sampled token
        away. The thinking state must not advance on them: the entropy there is
        not a decode step's entropy, and a mid-prompt argmax that happened to be
        `</think>` would silently end the row's thinking block before it ever
        generated. The model runner refreshes this from its discard mask every
        step, before the sampler runs; `step` leaves these rows untouched --
        state, stored mixture and forced token alike.
        """
        self._skip_rows = frozenset(rows)

    def _tracked_rows(self) -> list[int]:
        """Row indices still inside their thinking block, in ascending order."""
        return sorted(
            i
            for i, s in self._state.items()
            if s["in_thinking"] and i not in self._skip_rows
        )

    def step(
        self,
        logits: torch.Tensor,
        temperature: torch.Tensor | None,
        top_k: torch.Tensor | None,
        top_p: torch.Tensor | None,
        min_p: torch.Tensor | None,
    ) -> torch.Tensor | None:
        """Advance every thinking row by one step.

        Computes each row's concept token, applies Cold Stop, and stores the
        mixtures for the model runner to feed back next step.

        Args:
            logits: ``[num_reqs, vocab]`` after penalties, before temperature.
            temperature: ``[num_reqs]`` or None (all-greedy batches).
            top_k: ``[num_reqs]`` or None.
            top_p: ``[num_reqs]`` or None.
            min_p: ``[num_reqs]`` or None.

        Returns:
            ``[num_reqs]`` of token ids to force, ``-1`` where nothing is forced.
            None when no row is thinking. Rows that Cold Stop fired on are forced
            to ``</think>`` so the trace closes properly.
        """
        from vllm.v1.sample.soft_thinking import compute_concept_tokens

        rows = self._tracked_rows()
        if not rows:
            return None

        idx = torch.tensor(rows, device=logits.device, dtype=torch.long)
        sub = lambda t: None if t is None else t[idx]  # noqa: E731

        temps = (
            sub(temperature)
            if temperature is not None
            else torch.ones(len(rows), device=logits.device)
        )
        # One k for the whole batch: torch.topk cannot take a per-row width. Rows
        # asking for less are trimmed below.
        soft_topk = max(self._state[r]["soft_topk"] for r in rows)
        concept = compute_concept_tokens(
            logits[idx],
            temperature=temps,
            soft_topk=soft_topk,
            top_k=sub(top_k),
            top_p=sub(top_p),
            min_p=sub(min_p),
        )
        concept = self._trim_to_per_row_topk(concept, rows)

        dev = logits.device
        in_thinking = torch.ones(len(rows), dtype=torch.bool, device=dev)
        low_steps = torch.tensor(
            [self._state[r]["low_entropy_steps"] for r in rows], device=dev
        )
        thresholds = torch.tensor(
            [self._state[r]["entropy_threshold"] for r in rows], device=dev
        )
        patience = torch.tensor([self._state[r]["patience"] for r in rows], device=dev)
        end_ids = torch.full(
            (len(rows),), self.think_end_token_id, device=dev, dtype=torch.long
        )

        in_thinking, low_steps, fired = update_cold_stop(
            in_thinking,
            low_steps,
            concept.entropy,
            concept.argmax_ids,
            thresholds,
            patience,
            end_ids,
        )

        still = in_thinking.tolist()
        steps = low_steps.tolist()
        for pos, r in enumerate(rows):
            self._state[r]["in_thinking"] = still[pos]
            self._state[r]["low_entropy_steps"] = steps[pos]

        # Only rows still thinking carry a concept token into the next step.
        for pos, r in enumerate(rows):
            if still[pos]:
                self._state[r]["concept_ids"] = concept.topk_ids[pos]
                self._state[r]["concept_probs"] = concept.topk_probs[pos]
            else:
                self._state[r].pop("concept_ids", None)
                self._state[r].pop("concept_probs", None)

        forced = torch.full((logits.shape[0],), -1, device=dev, dtype=torch.long)
        # The thinking phase does not commit to a token, so what lands in the
        # trace is bookkeeping: record the argmax, as the reference
        # implementation does. Sampling it instead would let the trace disagree
        # with the </think> detection below, which reads the argmax.
        forced[idx] = self._recorded_ids(concept, rows)
        forced[idx[fired]] = self.think_end_token_id
        return forced

    def _recorded_ids(
        self, concept: ConceptTokens, rows: list[int]
    ) -> torch.Tensor:
        """The argmax, but never a stop token while the row is still thinking.

        vLLM ends a sequence as soon as a stop id is produced, and the reference
        implementation deliberately does not let that happen mid-thinking -- "a
        stop token is not meaningful while the row is still consuming continuous
        inputs". Recording one here would cut the sample off before it ever
        reaches its answer. The next candidate in the same top-k is used instead;
        the mixture fed back is untouched, so only the trace changes.

        The stop ids of every tracked row are pooled rather than applied per
        row. They are the model's own terminators and so identical in practice,
        and the cost of pooling is at worst a different high-probability token in
        one row's trace.
        """
        stop_ids = set().union(*(self._state[r]["stop_ids"] for r in rows))
        if not stop_ids:
            return concept.argmax_ids
        stops = torch.tensor(
            sorted(stop_ids), device=concept.topk_ids.device, dtype=torch.long
        )
        is_stop = (concept.topk_ids.unsqueeze(-1) == stops).any(dim=-1)
        # argmax over a bool picks the first True: the first non-stop candidate.
        # An all-stop row falls back to index 0, which cannot arise in practice.
        first_ok = (~is_stop).to(torch.int8).argmax(dim=1, keepdim=True)
        return concept.topk_ids.gather(1, first_ok).squeeze(1)

    def step_from_metadata(
        self, logits: torch.Tensor, sampling_metadata
    ) -> torch.Tensor | None:
        """:meth:`step`, reading the filters off the sampling metadata.

        ``min_p`` is not a metadata field: it is applied by an argmax-invariant
        logits processor further down, so it has to be read off that processor.
        """
        return self.step(
            logits,
            sampling_metadata.temperature,
            sampling_metadata.top_k,
            sampling_metadata.top_p,
            _min_p_from_metadata(sampling_metadata, logits.shape[0]),
        )

    def _trim_to_per_row_topk(
        self, concept: ConceptTokens, rows: list[int]
    ) -> ConceptTokens:
        """Zero the weights past each row's own soft_topk and renormalise.

        Only does work when rows disagree about k, which is the uncommon case.
        """
        ks = [self._state[r]["soft_topk"] for r in rows]
        if len(set(ks)) == 1:
            return concept
        width = concept.topk_ids.shape[1]
        k_t = torch.tensor(ks, device=concept.topk_ids.device).unsqueeze(-1)
        positions = torch.arange(width, device=concept.topk_ids.device)
        keep = positions.unsqueeze(0) < k_t
        probs = concept.topk_probs * keep
        probs = probs / (probs.sum(dim=-1, keepdim=True) + 1e-10)
        return ConceptTokens(
            topk_ids=concept.topk_ids,
            topk_probs=probs,
            entropy=concept.entropy,
            argmax_ids=concept.argmax_ids,
        )

    def concept_tokens_for_rows(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None:
        """The mixtures to feed back, as ``(row_indices, topk_ids, topk_probs)``.

        Row indices are into the sampling batch; the model runner maps them onto
        the positions it is about to embed. None before the first step, or once
        every row has left its thinking block.
        """
        rows = [r for r in sorted(self._state) if "concept_ids" in self._state[r]]
        if not rows:
            return None
        ids = torch.stack([self._state[r]["concept_ids"] for r in rows])
        probs = torch.stack([self._state[r]["concept_probs"] for r in rows])
        return torch.tensor(rows, device=ids.device, dtype=torch.long), ids, probs
