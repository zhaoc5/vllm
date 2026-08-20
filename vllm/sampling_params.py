# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Sampling parameters for text generation."""

import copy
import json as json_mod
import math
from dataclasses import field
from enum import Enum, IntEnum
from functools import cached_property
from typing import Annotated, Any

import msgspec
from pydantic import BeforeValidator
from pydantic.dataclasses import dataclass

import vllm.envs as envs
from vllm.config import ModelConfig, SpeculativeConfig, StructuredOutputsConfig
from vllm.exceptions import VLLMValidationError
from vllm.logger import init_logger
from vllm.tokenizers import TokenizerLike
from vllm.utils.mistral import is_mistral_tokenizer
from vllm.v1.serial_utils import PydanticMsgspecMixin

logger = init_logger(__name__)

_SAMPLING_EPS = 1e-5
_MAX_TEMP = 1e-2

MAX_LOGPROB_TOKEN_IDS = 128
"""Upper bound on `SamplingParams.logprob_token_ids` list length. Must match
the per-request row width allocated by the sampler's `LogprobTokenIdsState`."""


def _verify_num_sequences(value: int, parameter_name: str) -> None:
    if not isinstance(value, int):
        raise VLLMValidationError(
            f"{parameter_name} must be an int, but is of type {type(value)}"
        )
    if value < 1:
        raise VLLMValidationError(f"{parameter_name} must be at least 1, got {value}.")
    max_n = envs.VLLM_MAX_N_SEQUENCES
    if value > max_n:
        raise VLLMValidationError(
            f"{parameter_name} must be at most {max_n}, got {value}. "
            "To increase this limit, set the VLLM_MAX_N_SEQUENCES "
            "environment variable."
        )


def validate_thinking_token_budget(value: int | float | bool | None) -> int | None:
    """Validate ``thinking_token_budget``; return ``None`` if unset."""
    if value is None:
        return None
    if isinstance(value, (bool, float)) or not isinstance(value, int):
        raise VLLMValidationError(
            "`thinking_token_budget` must be a non-negative integer "
            "or -1 for unlimited.",
            parameter="thinking_token_budget",
            value=value,
        )
    if value == -1:
        return None
    if value < 0:
        raise VLLMValidationError(
            "`thinking_token_budget` must be a non-negative integer "
            "or -1 for unlimited.",
            parameter="thinking_token_budget",
            value=value,
        )
    return value


ThinkingTokenBudget = Annotated[
    int | None,
    BeforeValidator(validate_thinking_token_budget),
]


class SamplingType(IntEnum):
    GREEDY = 0
    RANDOM = 1
    RANDOM_SEED = 2


# maybe make msgspec?
@dataclass
class StructuredOutputsParams:
    # One of these fields will be used to build a logit processor.
    json: str | dict | None = None
    regex: str | None = None
    choice: list[str] | None = None
    grammar: str | None = None
    json_object: bool | None = None
    # These are other options that can be set.
    disable_any_whitespace: bool = False
    disable_additional_properties: bool = False
    whitespace_pattern: str | None = None
    structural_tag: str | None = None

    _backend: str | None = field(default=None, init=False)
    """CAUTION: Should only be set by Processor._validate_structured_output"""
    _backend_was_auto: bool = field(default=False, init=False)
    """CAUTION: Should only be set by Processor._validate_structured_output"""

    def __post_init__(self):
        """Validate that some fields are mutually exclusive."""
        count = sum(
            [
                self.json is not None,
                self.regex is not None,
                self.choice is not None,
                self.grammar is not None,
                self.json_object is not None,
                self.structural_tag is not None,
            ]
        )
        if count > 1:
            raise VLLMValidationError(
                "You can only use one kind of structured outputs constraint "
                f"but multiple are specified: {self.__dict__}"
            )
        if count < 1:
            raise VLLMValidationError(
                "You must use one kind of structured outputs constraint "
                f"but none are specified: {self.__dict__}"
            )

    def all_constraints_none(self) -> bool:
        """
        Returns True if all structured-output constraint fields are None.
        """
        return all(
            getattr(self, field) is None
            for field in (
                "json",
                "regex",
                "choice",
                "grammar",
                "json_object",
                "structural_tag",
            )
        )

    def all_non_structural_tag_constraints_none(self) -> bool:
        """
        Returns True if all structured-output constraint fields are None.
        """
        return all(
            getattr(self, field) is None
            for field in (
                "json",
                "regex",
                "choice",
                "grammar",
                "json_object",
            )
        )


@dataclass
class RepetitionDetectionParams:
    """Parameters for detecting repetitive N-gram patterns in output tokens."""

    max_pattern_size: int = 0
    """Maximum size of N-gram pattern to detect for sequence repetition.
    Set to 0 to disable. Must be used together with min_count."""

    min_pattern_size: int = 0
    """Minimum N-gram pattern size to check for sequence repetition.
    If set to 0, it defaults to 1.
    Must be <= max_pattern_size."""

    min_count: int = 0
    """Minimum number of times an N-gram pattern must repeat to trigger
    detection. Must be >= 2. Example: 3 for detecting a phrase repeated
    3 times. Must be used together with max_pattern_size."""

    def __post_init__(self):
        if (
            self.max_pattern_size < 0
            or self.min_pattern_size < 0
            or self.min_pattern_size > self.max_pattern_size
        ):
            raise VLLMValidationError(
                "max_pattern_size, min_pattern_size must be >=0, "
                "with min_pattern_size <= max_pattern_size. "
                "Set both to 0 to disable repetitive pattern detection."
            )
        if self.max_pattern_size > 0 and self.min_count < 2:
            raise VLLMValidationError(
                "min_count must be >= 2 to detect repetitive patterns "
                "in engine output. If you do not wish to detect repetitive "
                "patterns, set max_pattern_size to 0."
            )


class RequestOutputKind(Enum):
    # Return entire output so far in every RequestOutput
    CUMULATIVE = 0
    # Return only deltas in each RequestOutput
    DELTA = 1
    # Do not return intermediate RequestOutput
    FINAL_ONLY = 2


def _is_non_tekken_mistral(tokenizer: TokenizerLike) -> bool:
    return is_mistral_tokenizer(tokenizer) and not tokenizer.is_tekken


def _get_llg_tokenizer(tokenizer: TokenizerLike) -> Any:
    return tokenizer.llg_tokenizer if is_mistral_tokenizer(tokenizer) else None


class SamplingParams(
    PydanticMsgspecMixin,
    msgspec.Struct,
    omit_defaults=True,  # type: ignore[call-arg]
    # required for @cached_property.
    dict=True,
):  # type: ignore[call-arg]
    """Sampling parameters for text generation.

    Overall, we follow the sampling parameters from the OpenAI text completion
    API (https://platform.openai.com/docs/api-reference/completions/create).
    In addition, we support beam search, which is not supported by OpenAI.
    """

    n: int = 1
    """Number of outputs to return for the given prompt request.

    The maximum allowed value is controlled by the ``VLLM_MAX_N_SEQUENCES``
    environment variable (default: 16384).

    NOTE:
        `AsyncLLM` streams outputs by default. When `n > 1`, all `n` outputs
        are generated and streamed cumulatively per request. To see all `n`
        outputs upon completion, use `output_kind=RequestOutputKind.FINAL_ONLY`
        in `SamplingParams`."""
    presence_penalty: float = 0.0
    """Penalizes new tokens based on whether they appear in the generated text
    so far. Values > 0 encourage the model to use new tokens, while values < 0
    encourage the model to repeat tokens."""
    frequency_penalty: float = 0.0
    """Penalizes new tokens based on their frequency in the generated text so
    far. Values > 0 encourage the model to use new tokens, while values < 0
    encourage the model to repeat tokens."""
    repetition_penalty: float = 1.0
    """Penalizes new tokens based on whether they appear in the prompt and the
    generated text so far. Values > 1 encourage the model to use new tokens,
    while values < 1 encourage the model to repeat tokens."""
    temperature: float = 1.0
    """Controls the randomness of the sampling. Lower values make the model
    more deterministic, while higher values make the model more random. Zero
    means greedy sampling."""
    top_p: float = 1.0
    """Controls the cumulative probability of the top tokens to consider. Must
    be in (0, 1]. Set to 1 to consider all tokens."""
    top_k: int = 0
    """Controls the number of top tokens to consider. Set to 0 (or -1) to
    consider all tokens."""
    min_p: float = 0.0
    """Represents the minimum probability for a token to be considered,
    relative to the probability of the most likely token. Must be in [0, 1].
    Set to 0 to disable this."""
    # begin of soft thinking
    soft_thinking: bool = False
    """Soft Thinking (arXiv:2505.15778). While the model is inside its thinking block,
    feed back the probability-weighted mixture of the top-`soft_topk` token embeddings
    rather than committing to a discrete token. The answer that follows `</think>` is
    sampled normally. The other `soft_*` fields do nothing unless this is set.

    Stop *token ids* are kept out of the recorded trace while a row is thinking, so
    they cannot end it early; `stop` *strings* are matched by the detokenizer against
    that trace and would still cut it mid-thought -- reasoning text hits strings like
    "\n\n" constantly -- so they are refused together with this flag."""
    soft_topk: int = 10
    """Number of tokens kept in the concept-token mixture."""
    soft_entropy_threshold: float = 0.01
    """Cold Stop: the entropy, in nats, below which a step counts as
    low-entropy. Measured on the full post-temperature distribution, never on
    the top_k/top_p-filtered one."""
    soft_patience: int = 256
    """Cold Stop: consecutive low-entropy steps before `</think>` is forced. The counter
    resets on any step above the threshold, so this counts a run, not a total."""
    # end of soft thinking
    # begin of swir
    swir: bool = False
    """SwiReasoning (the swir method): switch each row between soft feedback (the
    full-vocabulary probability-weighted embedding mixture) and ordinary discrete
    feedback on the entropy trend of the raw distribution. Sampling and the recorded
    trace stay ordinary; only the embedding fed back changes. The other `swir_*`
    fields do nothing unless this is set."""
    swir_alpha: float = 1.0
    """Blend floor toward `<think>` when a row switches back to soft; ramps to 1
    over the generation."""
    swir_beta: float = 0.7
    """Blend floor toward `</think>` when a row switches to normal; ramps to 1
    over the generation."""
    swir_window: int = 512
    """Steps a row must stay in normal mode before it may switch back to soft."""
    swir_max_switch_count: int | None = None
    """Soft->normal switches before convergence is forced: after this many the row
    is fed the convergence tokens, after twice this many the termination tokens and
    a cutoff `swir_termination_max_tokens` later. None disables forcing."""
    swir_termination_max_tokens: int = 32
    """Tokens allowed after the termination phrase before the row is cut off."""
    swir_math_token_ids: list[int] | None = None
    """Token ids fed back discretely even in soft mode -- math notation drifts
    badly when consumed as a mixture. Tokenizer-dependent, so the caller supplies
    them."""
    swir_convergence_token_ids: list[int] | None = None
    """Tokenization of the convergence phrase (typically `</think>`)."""
    swir_termination_token_ids: list[int] | None = None
    """Tokenization of the termination phrase."""
    swir_linebreak_token_id: int | None = None
    """The line-break token blended into the very first soft step, as the
    reference implementation does."""
    # end of swir
    # begin of selar
    selar: bool = False
    """SeLaR (the selar method): gate each step on the entropy of the renormalised
    top-`selar_topk` head of the sampling distribution; gated steps feed back the
    head's embedding mixture displaced away from the top-1 embedding instead of the
    sampled token's embedding. Sampling and the recorded trace stay ordinary. The
    other `selar_*` fields do nothing unless this is set."""
    selar_topk: int = 5
    """Head size for the gate's entropy and the latent mixture."""
    selar_entropy_threshold: float = 0.3
    """Normalised-entropy threshold in [0, 1] above which a step feeds back the
    latent input."""
    selar_contrastive_weight: float = 1.0
    """Scale of the contrastive displacement away from the top-1 embedding;
    multiplied by the gate signal, so more uncertain steps are pushed harder."""
    selar_math_token_ids: list[int] | None = None
    """Token ids fed back discretely even when the gate is open. Tokenizer-
    dependent, so the caller supplies them."""
    # end of selar
    seed: int | None = None
    """Random seed to use for the generation."""
    stop: str | list[str] | None = None
    """String(s) that stop the generation when they are generated. The returned
    output will not contain the stop strings."""
    stop_token_ids: list[int] | None = None
    """Token IDs that stop the generation when they are generated. The returned
    output will contain the stop tokens unless the stop tokens are special
    tokens."""
    ignore_eos: bool = False
    """Whether to ignore the EOS token and continue generating
    tokens after the EOS token is generated."""
    max_tokens: int | None = 16
    """Maximum number of tokens to generate per output sequence."""
    min_tokens: int = 0
    """Minimum number of tokens to generate per output sequence before EOS or
    `stop_token_ids` can be generated"""
    logprobs: int | None = None
    """Number of log probabilities to return per output token. When set to
    `None`, no probability is returned. If set to a non-`None` value, the
    result includes the log probabilities of the specified number of most
    likely tokens, as well as the chosen tokens. Note that the implementation
    follows the OpenAI API: The API will always return the log probability of
    the sampled token, so there may be up to `logprobs+1` elements in the
    response. When set to -1, return all `vocab_size` log probabilities."""
    prompt_logprobs: int | None = None
    """Number of log probabilities to return per prompt token.
    When set to -1, return all `vocab_size` log probabilities."""
    logprob_token_ids: list[int] | None = None
    """Specific token IDs to return logprobs for. More efficient than
    logprobs=-1 when you only need logprobs for a small set of tokens.
    When set, logprobs for exactly these token IDs will be returned,
    in addition to the sampled token. This is useful for scoring tasks
    where you want to compare probabilities of specific label tokens."""
    flat_logprobs: bool = False
    """Whether to return logprobs in flatten format (i.e. FlatLogprob)
    for better performance.
    NOTE: GC costs of FlatLogprobs is significantly smaller than
    list[dict[int, Logprob]]. After enabled, PromptLogprobs and
    SampleLogprobs would populated as FlatLogprobs."""
    # NOTE: This parameter is only exposed at the engine level for now.
    # It is not exposed in the OpenAI API server, as the OpenAI API does
    # not support returning only a list of token IDs.
    detokenize: bool = True
    """Whether to detokenize the output."""
    skip_special_tokens: bool = True
    """Whether to skip special tokens in the output."""
    spaces_between_special_tokens: bool = True
    """Whether to add spaces between special tokens in the output."""
    include_stop_str_in_output: bool = False
    """Whether to include the stop strings in output text."""
    output_kind: RequestOutputKind = RequestOutputKind.CUMULATIVE
    stream_interval: int | None = None
    """Number of newly generated tokens to batch into each streamed
    `RequestOutput`. Raises the interval above the engine-level
    `--stream-interval`. Values below engine setting are clamped up to it.
    The first and final outputs are always emitted immediately."""
    skip_clone: bool = False
    """Internal flag indicating that this SamplingParams instance is safe to
    reuse without cloning. When True, clone() will return self without
    performing a deep copy. This should only be set when the params object
    is guaranteed to be dedicated to a single request and won't be modified
    in ways that would affect other uses."""

    # The below fields are not supposed to be used as an input.
    # They are set in post_init.
    output_text_buffer_length: int = 0
    _eos_token_id: int | None = None
    _all_stop_token_ids: set[int] = msgspec.field(default_factory=set)

    # Fields used to construct logits processors
    structured_outputs: StructuredOutputsParams | None = None
    """Parameters for configuring structured outputs."""
    logit_bias: dict[int, float] | None = None
    """If provided, the engine will construct a logits processor that applies
    these logit biases."""
    allowed_token_ids: list[int] | None = None
    """If provided, the engine will construct a logits processor which only
    retains scores for the given token ids."""
    extra_args: dict[str, Any] | None = None
    """Arbitrary additional args, that can be used by custom sampling
    implementations, plugins, etc. Not used by any in-tree sampling
    implementations."""
    routed_experts_prompt_start: int = 0
    """When enable_return_routed_experts is active, skip the first
    routed_experts_prompt_start prompt tokens from the returned routing
    data. In multi-turn agent scenarios, set this to the length of the
    already-returned prefix to avoid duplicating routing for prompt tokens
    covered by earlier turns. Default 0 returns routing for all prompt
    tokens."""

    # Fields used for bad words
    bad_words: list[str] | None = None
    """Words that are not allowed to be generated. More precisely, only the
    last token of a corresponding token sequence is not allowed when the next
    generated token can complete the sequence."""
    _bad_words_token_ids: list[list[int]] | None = None

    skip_reading_prefix_cache: bool | None = None
    thinking_token_budget: int | None = None
    """Maximum number of tokens allowed for thinking operations."""

    repetition_detection: RepetitionDetectionParams | None = None
    """Parameters for detecting repetitive N-gram patterns in output tokens.
    If such repetition is detected, generation will be ended early. LLMs can
    sometimes generate repetitive, unhelpful token patterns, stopping only
    when they hit the maximum output length (e.g. 'abcdabcdabcd...' or
    '\\emoji \\emoji \\emoji ...'). This feature can detect such behavior
    and terminate early, saving time and tokens."""

    @staticmethod
    def from_optional(
        n: int | None = 1,
        presence_penalty: float | None = 0.0,
        frequency_penalty: float | None = 0.0,
        repetition_penalty: float | None = 1.0,
        temperature: float | None = 1.0,
        top_p: float | None = 1.0,
        top_k: int = 0,
        min_p: float = 0.0,
        soft_thinking: bool = False,
        soft_topk: int = 10,
        soft_entropy_threshold: float = 0.01,
        soft_patience: int = 256,
        swir: bool = False,
        swir_alpha: float = 1.0,
        swir_beta: float = 0.7,
        swir_window: int = 512,
        swir_max_switch_count: int | None = None,
        swir_termination_max_tokens: int = 32,
        swir_math_token_ids: list[int] | None = None,
        swir_convergence_token_ids: list[int] | None = None,
        swir_termination_token_ids: list[int] | None = None,
        swir_linebreak_token_id: int | None = None,
        selar: bool = False,
        selar_topk: int = 5,
        selar_entropy_threshold: float = 0.3,
        selar_contrastive_weight: float = 1.0,
        selar_math_token_ids: list[int] | None = None,
        seed: int | None = None,
        stop: str | list[str] | None = None,
        stop_token_ids: list[int] | None = None,
        bad_words: list[str] | None = None,
        thinking_token_budget: int | None = None,
        include_stop_str_in_output: bool = False,
        ignore_eos: bool = False,
        max_tokens: int | None = 16,
        min_tokens: int = 0,
        logprobs: int | None = None,
        prompt_logprobs: int | None = None,
        detokenize: bool = True,
        skip_special_tokens: bool = True,
        spaces_between_special_tokens: bool = True,
        output_kind: RequestOutputKind = RequestOutputKind.CUMULATIVE,
        stream_interval: int | None = None,
        structured_outputs: StructuredOutputsParams | None = None,
        logit_bias: dict[int, float] | dict[str, float] | None = None,
        allowed_token_ids: list[int] | None = None,
        extra_args: dict[str, Any] | None = None,
        skip_clone: bool = False,
        repetition_detection: RepetitionDetectionParams | None = None,
        logprob_token_ids: list[int] | None = None,
        routed_experts_prompt_start: int = 0,
    ) -> "SamplingParams":
        if logit_bias is not None:
            # Fast path uses a dict comprehension; on failure we iterate once
            # to identify the exact offending entry for the error message.
            try:
                logit_bias = {
                    int(token): min(100.0, max(-100.0, bias))
                    for token, bias in logit_bias.items()
                }
            except (ValueError, TypeError):
                invalid_keys = []
                converted_logit_bias = {}
                for token, bias in logit_bias.items():
                    try:
                        token_id = int(token)
                    except (ValueError, TypeError):
                        invalid_keys.append(token)
                        continue
                    converted_logit_bias[token_id] = min(100.0, max(-100.0, bias))
                if invalid_keys:
                    raise VLLMValidationError(
                        f"logit_bias contains key(s) that cannot be "
                        f"converted to integer token IDs: {invalid_keys!r}",
                        parameter="logit_bias",
                        value=invalid_keys,
                    ) from None
                logit_bias = converted_logit_bias

        return SamplingParams(
            n=1 if n is None else n,
            presence_penalty=0.0 if presence_penalty is None else presence_penalty,
            frequency_penalty=0.0 if frequency_penalty is None else frequency_penalty,
            repetition_penalty=1.0
            if repetition_penalty is None
            else repetition_penalty,
            temperature=1.0 if temperature is None else temperature,
            top_p=1.0 if top_p is None else top_p,
            top_k=top_k,
            min_p=min_p,
            soft_thinking=soft_thinking,
            soft_topk=soft_topk,
            soft_entropy_threshold=soft_entropy_threshold,
            soft_patience=soft_patience,
            swir=swir,
            swir_alpha=swir_alpha,
            swir_beta=swir_beta,
            swir_window=swir_window,
            swir_max_switch_count=swir_max_switch_count,
            swir_termination_max_tokens=swir_termination_max_tokens,
            swir_math_token_ids=swir_math_token_ids,
            swir_convergence_token_ids=swir_convergence_token_ids,
            swir_termination_token_ids=swir_termination_token_ids,
            swir_linebreak_token_id=swir_linebreak_token_id,
            selar=selar,
            selar_topk=selar_topk,
            selar_entropy_threshold=selar_entropy_threshold,
            selar_contrastive_weight=selar_contrastive_weight,
            selar_math_token_ids=selar_math_token_ids,
            seed=seed,
            stop=stop,
            stop_token_ids=stop_token_ids,
            bad_words=bad_words,
            thinking_token_budget=thinking_token_budget,
            include_stop_str_in_output=include_stop_str_in_output,
            ignore_eos=ignore_eos,
            max_tokens=max_tokens,
            min_tokens=min_tokens,
            logprobs=logprobs,
            prompt_logprobs=prompt_logprobs,
            logprob_token_ids=logprob_token_ids,
            detokenize=detokenize,
            skip_special_tokens=skip_special_tokens,
            spaces_between_special_tokens=spaces_between_special_tokens,
            output_kind=output_kind,
            stream_interval=stream_interval,
            structured_outputs=structured_outputs,
            logit_bias=logit_bias,
            allowed_token_ids=allowed_token_ids,
            extra_args=extra_args,
            skip_clone=skip_clone,
            repetition_detection=repetition_detection,
            routed_experts_prompt_start=routed_experts_prompt_start,
        )

    def __post_init__(self) -> None:
        if 0 < self.temperature < _MAX_TEMP:
            logger.warning(
                "temperature %s is less than %s, which may cause numerical "
                "errors nan or inf in tensors. We have maxed it out to %s.",
                self.temperature,
                _MAX_TEMP,
                _MAX_TEMP,
            )
            self.temperature = max(self.temperature, _MAX_TEMP)

        if self.seed == -1:
            self.seed = None

        self.thinking_token_budget = validate_thinking_token_budget(
            self.thinking_token_budget
        )

        if self.stop is None:
            self.stop = []
        elif isinstance(self.stop, str):
            self.stop = [self.stop]

        if self.stop_token_ids is None:
            self.stop_token_ids = []
        else:
            self.stop_token_ids = list(dict.fromkeys(self.stop_token_ids))

        if self.bad_words is None:
            self.bad_words = []
        else:
            self.bad_words = list(dict.fromkeys(self.bad_words))

        if self.logprobs is True:
            self.logprobs = 1

        if self.prompt_logprobs is True:
            self.prompt_logprobs = 1

        # Number of characters to hold back for stop string evaluation
        # until sequence is finished.
        if self.stop and not self.include_stop_str_in_output:
            self.output_text_buffer_length = max(len(s) for s in self.stop) - 1

        self._verify_args()

        if self.temperature < _SAMPLING_EPS:
            # Zero temperature means greedy sampling.
            self.top_p = 1.0
            self.top_k = 0
            self.min_p = 0.0
            self._verify_greedy_sampling()

        # eos_token_id is added to this by the engine
        self._all_stop_token_ids.update(self.stop_token_ids)

        if self.skip_reading_prefix_cache is None:
            # If prefix caching is enabled,
            # the output of prompt logprobs may less than n_prompt_tokens,
            # we need to skip reading cache at this request.
            self.skip_reading_prefix_cache = self.prompt_logprobs is not None

    def _verify_args(self) -> None:
        _verify_num_sequences(self.n, "n")
        # begin of soft thinking
        if self.soft_thinking:
            if self.soft_topk < 1:
                raise VLLMValidationError(
                    f"soft_topk must be at least 1, got {self.soft_topk}.",
                    parameter="soft_topk",
                    value=self.soft_topk,
                )
            if self.soft_patience < 1:
                raise VLLMValidationError(
                    f"soft_patience must be at least 1, got {self.soft_patience}.",
                    parameter="soft_patience",
                    value=self.soft_patience,
                )
            if self.soft_entropy_threshold < 0.0:
                raise VLLMValidationError(
                    "soft_entropy_threshold must be non-negative, got "
                    f"{self.soft_entropy_threshold}.",
                    parameter="soft_entropy_threshold",
                    value=self.soft_entropy_threshold,
                )
            if self.n != 1:
                # Each sequence carries its own thinking state and its own concept
                # token; n > 1 would need one per branch, which is not wired up.
                raise VLLMValidationError(
                    f"soft_thinking supports n=1 only, got n={self.n}.",
                    parameter="n",
                    value=self.n,
                )
            if self.stop:
                # Stop token ids are kept out of the recorded trace while a row
                # is thinking, but stop *strings* are matched by the detokenizer
                # against that trace and would end the request mid-thought.
                raise VLLMValidationError(
                    "soft_thinking does not support stop strings: the "
                    "detokenizer would match them against the thinking trace "
                    "and cut the request off mid-thought. Use stop_token_ids.",
                    parameter="stop",
                    value=self.stop,
                )
        # end of soft thinking
        # begin of swir
        if self.swir:
            if self.soft_thinking:
                raise VLLMValidationError(
                    "swir and soft_thinking are mutually exclusive: each "
                    "prescribes its own feedback embedding for every step.",
                    parameter="swir",
                    value=True,
                )
            if self.n != 1:
                raise VLLMValidationError(
                    f"swir supports n=1 only, got n={self.n}.",
                    parameter="n",
                    value=self.n,
                )
            if self.swir_window < 1:
                raise VLLMValidationError(
                    f"swir_window must be at least 1, got {self.swir_window}.",
                    parameter="swir_window",
                    value=self.swir_window,
                )
            if (self.swir_max_switch_count is not None
                    and self.swir_max_switch_count < 1):
                raise VLLMValidationError(
                    "swir_max_switch_count must be at least 1 when set, got "
                    f"{self.swir_max_switch_count}.",
                    parameter="swir_max_switch_count",
                    value=self.swir_max_switch_count,
                )
            if self.swir_max_switch_count is not None and not (
                self.swir_convergence_token_ids
                and self.swir_termination_token_ids
            ):
                raise VLLMValidationError(
                    "swir_max_switch_count needs swir_convergence_token_ids "
                    "and swir_termination_token_ids: they are what gets "
                    "injected when the switch budget runs out.",
                    parameter="swir_max_switch_count",
                    value=self.swir_max_switch_count,
                )
            if self.swir_linebreak_token_id is None:
                raise VLLMValidationError(
                    "swir requires swir_linebreak_token_id: the very first "
                    "soft step blends the mixture with the line-break "
                    "embedding, and the id is tokenizer-dependent.",
                    parameter="swir_linebreak_token_id",
                    value=None,
                )
        # end of swir
        # begin of selar
        if self.selar:
            if self.soft_thinking or self.swir:
                raise VLLMValidationError(
                    "selar, swir and soft_thinking are mutually exclusive: "
                    "each prescribes its own feedback embedding for every step.",
                    parameter="selar",
                    value=True,
                )
            if self.n != 1:
                raise VLLMValidationError(
                    f"selar supports n=1 only, got n={self.n}.",
                    parameter="n",
                    value=self.n,
                )
            if self.selar_topk < 2:
                raise VLLMValidationError(
                    "selar_topk must be at least 2: the gate's entropy ceiling "
                    f"is log(k), which is zero for k=1. Got {self.selar_topk}.",
                    parameter="selar_topk",
                    value=self.selar_topk,
                )
            if not 0.0 <= self.selar_entropy_threshold <= 1.0:
                raise VLLMValidationError(
                    "selar_entropy_threshold is a normalised entropy in "
                    f"[0, 1], got {self.selar_entropy_threshold}.",
                    parameter="selar_entropy_threshold",
                    value=self.selar_entropy_threshold,
                )
        # end of selar
        if not -2.0 <= self.presence_penalty <= 2.0:
            raise VLLMValidationError(
                f"presence_penalty must be in [-2, 2], got {self.presence_penalty}."
            )
        if not -2.0 <= self.frequency_penalty <= 2.0:
            raise VLLMValidationError(
                f"frequency_penalty must be in [-2, 2], got {self.frequency_penalty}."
            )
        if not math.isfinite(self.repetition_penalty):
            raise VLLMValidationError(
                "repetition_penalty must be a finite number, "
                f"got {self.repetition_penalty}."
            )
        if self.repetition_penalty <= 0.0:
            raise VLLMValidationError(
                "repetition_penalty must be greater than zero, got "
                f"{self.repetition_penalty}."
            )
        if not math.isfinite(self.temperature):
            raise VLLMValidationError(
                f"temperature must be a finite number, got {self.temperature}.",
                parameter="temperature",
                value=self.temperature,
            )
        if self.temperature < 0.0:
            raise VLLMValidationError(
                f"temperature must be non-negative, got {self.temperature}.",
                parameter="temperature",
                value=self.temperature,
            )
        if self.temperature > 2.0:
            raise VLLMValidationError(
                f"temperature must be in [0, 2], got {self.temperature}.",
                parameter="temperature",
                value=self.temperature,
            )
        if not 0.0 < self.top_p <= 1.0:
            raise VLLMValidationError(
                f"top_p must be in (0, 1], got {self.top_p}.",
                parameter="top_p",
                value=self.top_p,
            )
        # quietly accept -1 as disabled, but prefer 0
        if self.top_k < -1:
            raise VLLMValidationError(
                f"top_k must be 0 (disable), or at least 1, got {self.top_k}."
            )
        if not isinstance(self.top_k, int):
            raise VLLMValidationError(
                f"top_k must be an integer, got {type(self.top_k).__name__}"
            )
        if not 0.0 <= self.min_p <= 1.0:
            raise VLLMValidationError(f"min_p must be in [0, 1], got {self.min_p}.")
        if self.max_tokens is not None and self.max_tokens < 1:
            raise VLLMValidationError(
                f"max_tokens must be at least 1, got {self.max_tokens}.",
                parameter="max_tokens",
                value=self.max_tokens,
            )
        if self.min_tokens < 0:
            raise VLLMValidationError(
                f"min_tokens must be greater than or equal to 0, got {self.min_tokens}."
            )
        if self.max_tokens is not None and self.min_tokens > self.max_tokens:
            raise VLLMValidationError(
                f"min_tokens must be less than or equal to "
                f"max_tokens={self.max_tokens}, got {self.min_tokens}."
            )
        if self.stream_interval is not None and self.stream_interval < 1:
            raise VLLMValidationError(
                f"stream_interval must be at least 1, got {self.stream_interval}.",
                parameter="stream_interval",
                value=self.stream_interval,
            )
        if self.logprobs is not None and self.logprobs != -1 and self.logprobs < 0:
            raise VLLMValidationError(
                f"logprobs must be non-negative or -1, got {self.logprobs}.",
                parameter="logprobs",
                value=self.logprobs,
            )
        if (
            self.prompt_logprobs is not None
            and self.prompt_logprobs != -1
            and self.prompt_logprobs < 0
        ):
            raise VLLMValidationError(
                f"prompt_logprobs must be non-negative or -1, got "
                f"{self.prompt_logprobs}.",
                parameter="prompt_logprobs",
                value=self.prompt_logprobs,
            )
        assert isinstance(self.stop_token_ids, list)
        if not all(isinstance(st_id, int) for st_id in self.stop_token_ids):
            raise VLLMValidationError(
                f"stop_token_ids must contain only integers, got {self.stop_token_ids}."
            )
        assert isinstance(self.stop, list)
        if any(not stop_str for stop_str in self.stop):
            raise VLLMValidationError("stop cannot contain an empty string.")
        if self.stop and not self.detokenize:
            raise VLLMValidationError(
                "stop strings are only supported when detokenize is True. "
                "Set detokenize=True to use stop."
            )
        assert isinstance(self.bad_words, list)
        if any(not bad_word for bad_word in self.bad_words):
            raise VLLMValidationError(
                f"bad_words cannot contain an empty string. "
                f"Got bad_words={self.bad_words}"
            )

    def _verify_greedy_sampling(self) -> None:
        if self.n > 1:
            raise VLLMValidationError(
                f"n must be 1 when using greedy sampling, got {self.n}."
            )

    def update_from_generation_config(
        self,
        generation_config: dict[str, Any],
        eos_token_id: int | None = None,
    ) -> None:
        """Update if there are non-default values from generation_config"""
        if not self.ignore_eos:
            self._eos_token_id = eos_token_id

        if eos_token_id is not None:
            # Add the eos token id into the sampling_params to support
            # min_tokens processing.
            self._all_stop_token_ids.add(eos_token_id)

        # Update eos_token_id for generation
        if (eos_ids := generation_config.get("eos_token_id")) is not None:
            # it can be either int or list of int
            eos_ids = {eos_ids} if isinstance(eos_ids, int) else set(eos_ids)
            if eos_token_id is not None:
                # We don't need to include the primary eos_token_id in
                # stop_token_ids since it's handled separately for stopping
                # purposes.
                eos_ids.discard(eos_token_id)
            if eos_ids:
                self._all_stop_token_ids.update(eos_ids)
                if not self.ignore_eos:
                    assert self.stop_token_ids is not None
                    eos_ids.update(self.stop_token_ids)
                    self.stop_token_ids = list(eos_ids)

    def update_from_tokenizer(self, tokenizer: TokenizerLike) -> None:
        if not self.bad_words:
            return
        self._bad_words_token_ids = []
        max_num_bad_words = envs.VLLM_MAX_NUM_BAD_WORDS
        for bad_word in self.bad_words:
            # To prohibit words both at the beginning
            # and in the middle of text
            # (related to add_prefix_space tokenizer parameter)
            for add_prefix_space in [False, True]:
                prefix = " " if add_prefix_space else ""
                prompt = prefix + bad_word.lstrip()
                prompt_token_ids = tokenizer.encode(
                    text=prompt, add_special_tokens=False
                )

                # If no space at the beginning
                # or if prefix space produces a new word token
                if (not add_prefix_space) or (
                    add_prefix_space
                    and prompt_token_ids[0] != self._bad_words_token_ids[-1][0]
                    and len(prompt_token_ids) == len(self._bad_words_token_ids[-1])
                ):
                    self._bad_words_token_ids.append(prompt_token_ids)
                    if len(self._bad_words_token_ids) > max_num_bad_words:
                        raise VLLMValidationError(
                            f"Too many bad words after tokenization: "
                            f"{len(self._bad_words_token_ids)}. "
                            f"The max number is {max_num_bad_words}.",
                            parameter="bad_words",
                            value=self.bad_words,
                        )

        invalid_token_ids = [
            token_id
            for bad_words_token_ids in self._bad_words_token_ids
            for token_id in bad_words_token_ids
            if token_id < 0 or token_id > tokenizer.max_token_id
        ]
        if len(invalid_token_ids) > 0:
            raise VLLMValidationError(
                f"The model vocabulary size is {tokenizer.max_token_id + 1},"
                f" but the following tokens"
                f" were specified as bad: {invalid_token_ids}."
                f" All token id values should be integers satisfying:"
                f" 0 <= token_id <= {tokenizer.max_token_id}.",
                parameter="bad_words",
                value=self.bad_words,
            )

    @cached_property
    def sampling_type(self) -> SamplingType:
        if self.temperature < _SAMPLING_EPS:
            return SamplingType.GREEDY
        if self.seed is not None:
            return SamplingType.RANDOM_SEED
        return SamplingType.RANDOM

    @property
    def eos_token_id(self) -> int | None:
        return self._eos_token_id

    @property
    def all_stop_token_ids(self) -> set[int]:
        return self._all_stop_token_ids

    @property
    def bad_words_token_ids(self) -> list[list[int]] | None:
        # For internal use only. Backward compatibility not guaranteed
        return self._bad_words_token_ids

    @property
    def num_logprobs(self) -> int | None:
        """Number of sample logprobs to return per output token, or `None` if
        no sample logprobs were requested. Takes `logprob_token_ids` into
        account: when `logprobs` is unset but `logprob_token_ids` is set,
        returns `len(logprob_token_ids)`."""
        if self.logprobs is not None:
            return self.logprobs
        return len(self.logprob_token_ids) if self.logprob_token_ids else None

    def clone(self) -> "SamplingParams":
        """If skip_clone is True, uses shallow copy instead of deep copy."""
        if self.skip_clone:
            return copy.copy(self)

        return copy.deepcopy(self)

    def verify(
        self,
        model_config: ModelConfig,
        speculative_config: SpeculativeConfig | None,
        structured_outputs_config: StructuredOutputsConfig | None,
        tokenizer: TokenizerLike | None,
    ) -> None:
        self._validate_logprobs(model_config)
        self._validate_logit_bias(model_config)
        self._validate_logits_processors(model_config)
        self._validate_allowed_token_ids(tokenizer)
        self._validate_spec_decode(speculative_config)
        self._validate_diffusion(model_config)
        self._validate_structured_outputs(
            model_config, structured_outputs_config, tokenizer
        )

    def _validate_logprobs(self, model_config: ModelConfig) -> None:
        max_logprobs = model_config.max_logprobs
        if max_logprobs == -1:
            max_logprobs = model_config.get_vocab_size()

        # Validate sample logprobs.
        if num_logprobs := self.logprobs:
            if num_logprobs == -1:
                num_logprobs = model_config.get_vocab_size()
            if num_logprobs > max_logprobs:
                raise VLLMValidationError(
                    f"Requested sample logprobs of {num_logprobs}, "
                    f"which is greater than max allowed: {max_logprobs}",
                    parameter="logprobs",
                    value=num_logprobs,
                )

        # Validate logprob_token_ids.
        if self.logprob_token_ids is not None:
            n = len(self.logprob_token_ids)
            if n > MAX_LOGPROB_TOKEN_IDS:
                raise VLLMValidationError(
                    f"Requested logprob_token_ids of length {n}, "
                    f"which is greater than max allowed: {MAX_LOGPROB_TOKEN_IDS}",
                    parameter="logprob_token_ids",
                    value=n,
                )
            vocab_size = model_config.get_vocab_size()
            invalid_token_ids = [
                token_id
                for token_id in self.logprob_token_ids
                if token_id < 0 or token_id >= vocab_size
            ]
            if invalid_token_ids:
                raise VLLMValidationError(
                    f"token_id(s) {invalid_token_ids} in logprob_token_ids "
                    f"contain out-of-vocab token ids. Vocabulary size: "
                    f"{vocab_size}",
                    parameter="logprob_token_ids",
                    value=invalid_token_ids,
                )
            if self.logprobs is not None and self.logprobs != n:
                raise VLLMValidationError(
                    f"When both logprobs and logprob_token_ids are set, "
                    f"logprobs must equal len(logprob_token_ids). Got "
                    f"logprobs={self.logprobs}, len(logprob_token_ids)={n}.",
                    parameter="logprob_token_ids",
                    value=n,
                )

        # Validate prompt logprobs.
        if num_prompt_logprobs := self.prompt_logprobs:
            if num_prompt_logprobs == -1:
                num_prompt_logprobs = model_config.get_vocab_size()
            if num_prompt_logprobs > max_logprobs:
                raise VLLMValidationError(
                    f"Requested prompt logprobs of {num_prompt_logprobs}, "
                    f"which is greater than max allowed: {max_logprobs}",
                    parameter="prompt_logprobs",
                    value=num_prompt_logprobs,
                )

    def _validate_logit_bias(self, model_config: ModelConfig) -> None:
        """Validate logit_bias token IDs are within vocabulary range."""
        if not self.logit_bias:
            return

        vocab_size = model_config.get_vocab_size()
        invalid_token_ids = [
            token_id
            for token_id in self.logit_bias
            if token_id < 0 or token_id >= vocab_size
        ]

        if invalid_token_ids:
            raise VLLMValidationError(
                f"token_id(s) {invalid_token_ids} in logit_bias contain "
                f"out-of-vocab token ids. Vocabulary size: {vocab_size}",
                parameter="logit_bias",
                value=invalid_token_ids,
            )

    def _validate_logits_processors(self, model_config: ModelConfig) -> None:
        from vllm.v1.sample.logits_processor import (
            validate_logits_processors_parameters,
        )

        validate_logits_processors_parameters(model_config.logits_processors, self)

    def _validate_allowed_token_ids(self, tokenizer: TokenizerLike | None) -> None:
        allowed_token_ids = self.allowed_token_ids
        if allowed_token_ids is None:
            return

        if len(allowed_token_ids) == 0:
            raise VLLMValidationError(
                "allowed_token_ids is not None and empty!",
                parameter="allowed_token_ids",
                value=allowed_token_ids,
            )

        if tokenizer is not None:
            vocab_size = len(tokenizer)
            invalid_token_ids = [
                token_id
                for token_id in allowed_token_ids
                if token_id < 0 or token_id >= vocab_size
            ]
            if invalid_token_ids:
                raise VLLMValidationError(
                    "allowed_token_ids contains out-of-vocab token id!",
                    parameter="allowed_token_ids",
                    value=invalid_token_ids,
                )

    def _validate_spec_decode(
        self,
        speculative_config: SpeculativeConfig | None,
    ) -> None:
        if speculative_config is None:
            return

        # Adaptive verification compacts logits after the forward pass, while
        # compute_topk_scores uses the scheduled layout in cu_num_logits_np.
        # TODO(lucas): lift this restriction. The true boundaries exist on device as
        # cu_num_logits; cu_num_generated_tokens is only read host-side after
        # the logprobs D2H, so it could ride along on that copy.
        if (
            speculative_config.enable_adaptive_verification
            and self.num_logprobs is not None
        ):
            raise ValueError(
                "Output logprobs are not supported with DSpark confidence-based "
                "verification."
            )

        # Some sampling parameters are not yet compatible with spec decoding.
        if self.min_p > _SAMPLING_EPS or self.logit_bias:
            raise VLLMValidationError(
                "The min_p and logit_bias sampling parameters "
                "are not yet supported with speculative decoding."
            )

    def _validate_diffusion(self, model_config: ModelConfig) -> None:
        if not model_config.is_diffusion:
            return

        # Diffusion models denoise a whole canvas per step with a fixed
        # temperature schedule, so per-request sampling parameters are not
        # supported. Penalties are ignored by the sampler with a warning.
        if (
            self.temperature != 1.0
            or self.min_p > _SAMPLING_EPS
            or self.seed is not None
            or self.min_tokens > 0
            or self.logit_bias
            or self.bad_words
            or self.allowed_token_ids
        ):
            raise VLLMValidationError(
                "The temperature, min_p, seed, min_tokens, logit_bias, "
                "bad_words, and allowed_token_ids sampling parameters "
                "are not yet supported with diffusion models."
            )

    def _validate_structured_outputs(
        self,
        model_config: ModelConfig,
        structured_outputs_config: StructuredOutputsConfig | None,
        tokenizer: TokenizerLike | None,
    ) -> None:
        if structured_outputs_config is None or self.structured_outputs is None:
            return

        if model_config.is_diffusion:
            # Diffusion LLMs denoise a whole canvas of tokens in parallel
            # rather than sampling left-to-right, which the grammar FSM
            # requires. Without this check, requests fail mid-generation
            # with an FSM rejection (HTTP 500). See issue #45436.
            raise VLLMValidationError(
                "Structured outputs are not yet supported for diffusion "
                "language models. Remove the structured output constraint "
                "(e.g. `response_format`, `structured_outputs`) from the "
                "request."
            )

        if tokenizer is None:
            raise VLLMValidationError(
                "Structured outputs requires a tokenizer so it can't be used with 'skip_tokenizer_init'"  # noqa: E501
            )

        backend = structured_outputs_config.backend
        if _backend := self.structured_outputs._backend:
            # Request-level backend selection is not supported.
            # The values may differ if `params` is reused and was set
            # to a specific backend based on `auto` behavior in a previous
            # request. We remember that it was set as a result of `auto`
            # using the `_backend_was_auto` field set in the params.
            if backend != _backend and not (
                backend == "auto" and self.structured_outputs._backend_was_auto
            ):
                raise VLLMValidationError(
                    "Request-level structured output backend selection is not "
                    f"supported. The request specified '{_backend}', but vLLM "
                    f"was initialised with '{backend}'. This error can be "
                    "resolved by removing '_backend' from the request."
                )
        else:
            self.structured_outputs._backend = backend

        # Request content validation
        if (
            isinstance(self.structured_outputs.choice, list)
            and not self.structured_outputs.choice
        ):
            # It is invalid for choice to be an empty list
            raise VLLMValidationError(
                f"Choice '{self.structured_outputs.choice}' cannot be an empty list"  # noqa: E501
            )
        # Reject empty string grammar early to avoid engine-side crashes
        if (
            isinstance(self.structured_outputs.grammar, str)
            and self.structured_outputs.grammar.strip() == ""
        ):
            raise VLLMValidationError(
                "structured_outputs.grammar cannot be an empty string"
            )
        # Reject empty string json schema early to avoid engine-side crashes
        if (
            isinstance(self.structured_outputs.json, str)
            and self.structured_outputs.json.strip() == ""
        ):
            raise VLLMValidationError(
                "structured_outputs.json cannot be an empty string"
            )
        # Reject json_object=False early to avoid engine-side crashes
        if self.structured_outputs.json_object is False:
            raise VLLMValidationError(
                "structured_outputs.json_object must be True if set; omit "
                "structured_outputs to disable structured outputs"
            )
        # Reject a regex containing a NUL byte early, in every backend mode. A
        # NUL is never meaningful in a regex pattern and is not handled by the
        # regex-to-grammar conversion. Checked here, before backend selection,
        # so it is a clean 400 rather than a silent fallback in the default
        # "auto" mode.
        if self.structured_outputs.regex and "\x00" in self.structured_outputs.regex:
            raise VLLMValidationError(
                "structured_outputs.regex must not contain a NUL character ('\\x00')"
            )

        from vllm.v1.structured_output.backend_guidance import (
            has_guidance_unsupported_json_features,
            validate_guidance_grammar,
        )
        from vllm.v1.structured_output.backend_lm_format_enforcer import (
            validate_structured_output_request_lm_format_enforcer,
        )
        from vllm.v1.structured_output.backend_outlines import (
            validate_structured_output_request_outlines,
        )
        from vllm.v1.structured_output.backend_xgrammar import validate_xgrammar_grammar

        if backend.startswith("xgrammar"):
            # xgrammar with no fallback
            validate_xgrammar_grammar(self)
        elif backend.startswith("guidance"):
            if _is_non_tekken_mistral(tokenizer=tokenizer):
                raise VLLMValidationError(
                    "Non-tekken Mistral tokenizers are not supported for the 'guidance'"
                    " structured output backend. Please either use a more recent "
                    "Mistral model, the ['xgrammar', 'outlines'] "
                    "backends or tokenizer_mode='hf' instead."
                )
            # TODO: ideally we would have the LLTokenizer here as Lark syntax
            # allows <|special_token|> and similar, see
            # https://github.com/guidance-ai/llguidance/blob/main/docs/syntax.md#special-tokens
            # Without tokenizer these are disallowed in grammars.
            validate_guidance_grammar(
                self,
                tokenizer=_get_llg_tokenizer(tokenizer),
            )
        elif backend == "outlines":
            # outlines backend
            validate_structured_output_request_outlines(self)
        elif backend == "lm-format-enforcer":
            # lm format enforcer backend
            if is_mistral_tokenizer(tokenizer):
                raise VLLMValidationError(
                    "Mistral tokenizer is not supported for the 'lm-format-enforcer' "
                    "structured output backend. Please use ['xgrammar', 'outlines'] "
                    "backends or tokenizer_mode='hf' instead."
                )
            validate_structured_output_request_lm_format_enforcer(self)
        else:
            # NOTE: backend must be "auto" here, because we have
            # checked supported_backends above.
            # In this mode, we set opinionated defaults based on what we think
            # will satisfy the most use cases without having to worry about
            # this setting. We include fallback behavior here, but not with any
            # other setting where a specific backend was specified.
            try:
                validate_xgrammar_grammar(self)
                self.structured_outputs._backend = "xgrammar"
            except VLLMValidationError:
                # The request either failed validation
                # or includes some jsonschema feature(s) that
                # are not supported in xgrammar.

                skip_guidance = _is_non_tekken_mistral(tokenizer)

                # Check if schema has features unsupported by guidance
                so_params = self.structured_outputs
                if not skip_guidance and so_params.json:
                    if isinstance(so_params.json, str):
                        try:
                            schema = json_mod.loads(so_params.json)
                        except json_mod.JSONDecodeError as e:
                            raise VLLMValidationError(
                                "Invalid JSON grammar specification."
                            ) from e
                    else:
                        schema = so_params.json
                    skip_guidance = has_guidance_unsupported_json_features(schema)

                if skip_guidance:
                    # Fall back to outlines if the tokenizer is non-tekken Mistral or
                    # the schema contains features unsupported by guidance
                    validate_structured_output_request_outlines(self)
                    self.structured_outputs._backend = "outlines"
                else:
                    # Fall back to guidance by default.
                    validate_guidance_grammar(
                        self,
                        tokenizer=_get_llg_tokenizer(tokenizer),
                    )
                    self.structured_outputs._backend = "guidance"
            # Remember that this backend was set automatically
            self.structured_outputs._backend_was_auto = True

        # Run post-init validation. This is also important to ensure subsequent
        # roundtrip serialization/deserialization won't fail.
        self.structured_outputs.__post_init__()

    def __repr__(self) -> str:
        return (
            f"SamplingParams(n={self.n}, "
            f"presence_penalty={self.presence_penalty}, "
            f"frequency_penalty={self.frequency_penalty}, "
            f"repetition_penalty={self.repetition_penalty}, "
            f"temperature={self.temperature}, "
            f"top_p={self.top_p}, "
            f"top_k={self.top_k}, "
            f"min_p={self.min_p}, "
            f"soft_thinking={self.soft_thinking}, "
            f"soft_topk={self.soft_topk}, "
            f"soft_entropy_threshold={self.soft_entropy_threshold}, "
            f"soft_patience={self.soft_patience}, "
            f"swir={self.swir}, "
            f"swir_alpha={self.swir_alpha}, "
            f"swir_beta={self.swir_beta}, "
            f"swir_window={self.swir_window}, "
            f"swir_max_switch_count={self.swir_max_switch_count}, "
            f"selar={self.selar}, "
            f"selar_topk={self.selar_topk}, "
            f"selar_entropy_threshold={self.selar_entropy_threshold}, "
            f"seed={self.seed}, "
            f"stop={self.stop}, "
            f"stop_token_ids={self.stop_token_ids}, "
            f"bad_words={self.bad_words}, "
            f"thinking_token_budget={self.thinking_token_budget}, "
            f"include_stop_str_in_output={self.include_stop_str_in_output}, "
            f"ignore_eos={self.ignore_eos}, "
            f"max_tokens={self.max_tokens}, "
            f"min_tokens={self.min_tokens}, "
            f"logprobs={self.logprobs}, "
            f"prompt_logprobs={self.prompt_logprobs}, "
            f"skip_special_tokens={self.skip_special_tokens}, "
            "spaces_between_special_tokens="
            f"{self.spaces_between_special_tokens}, "
            f"structured_outputs={self.structured_outputs}, "
            f"extra_args={self.extra_args})"
        )

    @staticmethod
    def for_sampler_warmup() -> "SamplingParams":
        """Set parameters to exercise all sampler logic."""
        return SamplingParams(
            temperature=0.9,
            top_p=0.9,
            top_k=50,
            min_p=0.1,
            frequency_penalty=0.5,
            presence_penalty=0.5,
            repetition_penalty=1.2,
            min_tokens=2,
            logit_bias={0: -1.0, 1: 0.5},
            _bad_words_token_ids=[[0], [1, 2]],
            logprobs=5,
            prompt_logprobs=1,
        )


class BeamSearchParams(
    msgspec.Struct,
    omit_defaults=True,  # type: ignore[call-arg]
    # required for @cached_property.
    dict=True,
):  # type: ignore[call-arg]
    """Beam search parameters for text generation."""

    beam_width: int
    max_tokens: int
    ignore_eos: bool = False
    temperature: float = 0.0
    length_penalty: float = 1.0
    include_stop_str_in_output: bool = False
    structured_outputs: StructuredOutputsParams | None = None

    def __post_init__(self) -> None:
        _verify_num_sequences(self.beam_width, "beam_width")
