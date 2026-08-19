# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Soft Thinking: continuous-concept decoding inside the thinking block.

Soft Thinking (arXiv:2505.15778) does not commit to a discrete token while the
model is reasoning. Each step feeds back the probability-weighted mixture of
token embeddings -- the paper's *concept token*, ``e = sum_k p_k e(k)``. Once
``</think>`` is the argmax, or Cold Stop fires, the row reverts to ordinary
discrete decoding.

This module owns the arithmetic only. ``VocabParallelEmbedding.weighted_forward``
turns the mixture into an embedding; the per-request thinking state lives with
the model runner.

Two invariants decide whether the method works at all, both learned from the
reference HF implementation this mirrors:

* The mixture comes from the **filtered** distribution, the Cold Stop entropy
  from the **full** one. Entropy after top_p measures top_p: once the top token
  clears 0.95 the nucleus holds one token and the renormalised entropy is
  exactly 0. Over 1500 Qwen3.5-4B decoding steps the filtered entropy had
  p95 = 0.0000 under two sampling recipes, against 0.0724 and 0.0014 for the
  full distribution.
* Cold Stop's threshold and patience are tied to the sampling recipe. On those
  steps the longest sub-threshold run was 249 under the paper's recipe
  (temperature 0.6, presence_penalty 0.0) but 71 under Qwen3.5's own
  (temperature 1.0, presence_penalty 1.5), against a default patience of 256.
"""

from dataclasses import dataclass

import torch


@dataclass
class ConceptTokens:
    """One soft-thinking step, for the whole batch.

    Attributes:
        topk_ids: ``[num_reqs, soft_topk]`` ids of the mixture components.
        topk_probs: ``[num_reqs, soft_topk]`` weights, renormalised to sum to 1.
        entropy: ``[num_reqs]`` entropy of the full post-temperature
            distribution, in nats. Cold Stop thresholds this.
        argmax_ids: ``[num_reqs]`` most probable token. Recorded as the step's
            output so the trace stays readable and ``</think>`` detectable,
            while the embedding fed back is the mixture.
    """

    topk_ids: torch.Tensor
    topk_probs: torch.Tensor
    entropy: torch.Tensor
    argmax_ids: torch.Tensor


def _apply_filters(
    logits: torch.Tensor,
    top_k: torch.Tensor | None,
    top_p: torch.Tensor | None,
    min_p: torch.Tensor | None,
) -> torch.Tensor:
    """Mask with ``-inf`` in top_k, top_p, min_p order, as the reference does."""
    if top_k is not None:
        # Off is spelled two ways: <= 0 by convention, >= vocab by vLLM's
        # metadata (gpu_input_batch stores vocab_size for requests without
        # top_k). Both must cut nothing, and neither may widen max_k -- one
        # vocab_size row would otherwise turn torch.topk into a full sort.
        enabled = (top_k > 0) & (top_k < logits.shape[-1])
        if bool(enabled.any()):
            max_k = int(top_k[enabled].max().item())
            top_values, _ = torch.topk(logits, max_k, dim=-1)
            k_index = top_k.clamp(min=1, max=max_k) - 1
            kth = top_values.gather(-1, k_index.unsqueeze(-1))
            kth = torch.where(
                enabled.unsqueeze(-1), kth, torch.full_like(kth, float("-inf"))
            )
            logits = torch.where(logits < kth, float("-inf"), logits)

    if top_p is not None:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
        cumulative = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
        sorted_mask = cumulative > top_p.unsqueeze(-1)
        # Keep the token that crosses the threshold: the nucleus is never empty.
        sorted_mask[..., 1:] = sorted_mask[..., :-1].clone()
        sorted_mask[..., 0] = False
        mask = sorted_mask.scatter(-1, sorted_indices, sorted_mask)
        logits = logits.masked_fill(mask, float("-inf"))

    if min_p is not None:
        # Relative to the peak, not an absolute floor: on a flat distribution an
        # absolute 0.001 cuts nearly the whole vocabulary.
        probs = torch.softmax(logits, dim=-1)
        threshold = min_p.unsqueeze(-1) * probs.amax(dim=-1, keepdim=True)
        logits = torch.where(probs < threshold, float("-inf"), logits)

    return logits


def compute_concept_tokens(
    logits: torch.Tensor,
    temperature: torch.Tensor,
    soft_topk: int,
    top_k: torch.Tensor | None = None,
    top_p: torch.Tensor | None = None,
    min_p: torch.Tensor | None = None,
) -> ConceptTokens:
    """Build the concept token and the Cold Stop signal for one decode step.

    Args:
        logits: ``[num_reqs, vocab]`` after penalties, before temperature.
        temperature: ``[num_reqs]``. Values under the sampler's epsilon mean
            greedy; clamped to 1.0 here, which leaves the ranking untouched.
        soft_topk: mixture components to keep, clamped to the vocabulary.
        top_k: ``[num_reqs]`` per-request top_k, or None to skip it.
        top_p: ``[num_reqs]`` per-request top_p, or None to skip it.
        min_p: ``[num_reqs]`` per-request min_p, or None to skip it.

    Returns:
        The batch's :class:`ConceptTokens`.
    """
    logits = logits.to(torch.float32)
    # Callers differ on whether these arrive flat or already unsqueezed for
    # broadcasting -- min_p's logits processor stores [num_reqs, 1]. Flatten so
    # the unsqueeze below cannot add a second trailing axis.
    temperature = temperature.reshape(-1)
    top_k = None if top_k is None else top_k.reshape(-1)
    top_p = None if top_p is None else top_p.reshape(-1)
    min_p = None if min_p is None else min_p.reshape(-1)

    safe_temperature = torch.where(
        temperature < 1e-5, torch.ones_like(temperature), temperature
    )
    scaled = logits / safe_temperature.unsqueeze(-1)

    # Full distribution: what Cold Stop measures. Deliberately pre-filter.
    full_probs = torch.softmax(scaled, dim=-1)
    entropy = -(full_probs * torch.log(full_probs.clamp_min(1e-12))).sum(dim=-1)

    # Filtered distribution: what the mixture is drawn from.
    probs = torch.softmax(_apply_filters(scaled, top_k, top_p, min_p), dim=-1)

    k = min(soft_topk, probs.shape[-1])
    topk_probs, topk_ids = torch.topk(probs, k, dim=-1)
    topk_probs = topk_probs / (topk_probs.sum(dim=-1, keepdim=True) + 1e-10)

    return ConceptTokens(
        topk_ids=topk_ids,
        topk_probs=topk_probs,
        entropy=entropy,
        argmax_ids=topk_ids[:, 0],
    )


def update_cold_stop(
    in_thinking: torch.Tensor,
    low_entropy_steps: torch.Tensor,
    entropy: torch.Tensor,
    argmax_ids: torch.Tensor,
    entropy_threshold: torch.Tensor,
    patience: torch.Tensor,
    think_end_token_id: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Advance the per-request thinking state by one step.

    A row leaves the thinking phase either because ``</think>`` came out on its
    own or because Cold Stop fired. The caller emits ``</think>`` for the firing
    rows, so the trace is well-formed either way.

    Args:
        in_thinking: ``[num_reqs]`` bool, whether the row is still reasoning.
        low_entropy_steps: ``[num_reqs]`` consecutive sub-threshold steps so far.
        entropy: ``[num_reqs]`` from :func:`compute_concept_tokens`.
        argmax_ids: ``[num_reqs]`` from :func:`compute_concept_tokens`.
        entropy_threshold: ``[num_reqs]`` per-request Cold Stop threshold.
        patience: ``[num_reqs]`` per-request Cold Stop patience.
        think_end_token_id: ``[num_reqs]`` per-request ``</think>`` id.

    Returns:
        ``(in_thinking, low_entropy_steps, cold_stop_fired)``, all ``[num_reqs]``.
    """
    low = (entropy < entropy_threshold) & in_thinking
    # Reset on any step above the threshold: this counts a run, not a total.
    low_entropy_steps = torch.where(
        low, low_entropy_steps + 1, torch.zeros_like(low_entropy_steps)
    )
    fired = in_thinking & (low_entropy_steps >= patience)
    natural_end = in_thinking & (argmax_ids == think_end_token_id)
    in_thinking = in_thinking & ~(fired | natural_end)
    low_entropy_steps = torch.where(
        in_thinking, low_entropy_steps, torch.zeros_like(low_entropy_steps)
    )
    return in_thinking, low_entropy_steps, fired
