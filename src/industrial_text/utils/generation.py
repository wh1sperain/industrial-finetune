"""Explicit completion and bounded output-budget policy (no torch dependency)."""


def output_budgets(generation, input_tokens):
    available = generation["context_limit"] - input_tokens
    if available <= 0:
        raise ValueError("Input exhausts the model context")
    requested = [generation["max_new_tokens"], *generation.get("retry_max_new_tokens", [])]
    if any(n <= 0 for n in requested) or requested != sorted(set(requested)):
        raise ValueError("Output budgets must be positive and strictly increasing")
    return sorted({min(n, available) for n in requested})


def completion_status(ids, eos_ids, text, budget, answer_stop=None):
    eos_index = next((i for i, token in enumerate(ids) if token in eos_ids), None)
    if answer_stop and answer_stop in text:
        return "answer_end", False, eos_index
    if eos_index is not None:
        return "eos", False, eos_index
    if len(ids) >= budget:
        return "length", True, None
    # An unexplained short generation is incomplete, not successful termination.
    return "unexpected_stop", False, None
