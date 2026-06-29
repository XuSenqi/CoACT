"""Length reward for compression data preparation.

Within a rollout group, each completion that passes the action-quality
threshold ``theta`` gets a length reward

    R_length = 0.5 - (|C| - L_min) / (L_max - L_min)

normalized so the shortest valid completion receives +0.5, the longest gets
-0.5, and all equally-valid completions get 0.0. Completions whose
``action_reward`` is below ``theta`` receive 0.0.

This function is consumed only by offline-RL length-pair construction, which
uses the returned values for **within-group ordering** — the scale is fixed
(no alpha) since only sign and relative magnitude matter.
"""


def compute_length_reward(
    compressed_outputs: list[str],
    action_rewards: list[float],
    theta: float,
) -> list[float]:
    """Compute per-completion length rewards for one rollout group.

    Args:
        compressed_outputs: Group's compressed output strings.
        action_rewards: Corresponding action rewards.
        theta: Action-quality threshold; completions below it get 0.0.

    Returns:
        One length reward per input, aligned by index.
    """
    n = len(compressed_outputs)
    if n == 0:
        return []

    valid_lengths = [
        len(compressed_outputs[i]) for i, r in enumerate(action_rewards) if r >= theta
    ]
    if not valid_lengths:
        return [0.0] * n

    l_min = min(valid_lengths)
    l_max = max(valid_lengths)

    if l_max == l_min:
        # All valid completions same length — no length discrimination possible.
        return [0.0] * n

    span = l_max - l_min
    rewards: list[float] = []
    for i, ar in enumerate(action_rewards):
        if ar < theta:
            rewards.append(0.0)
        else:
            rewards.append(0.5 - (len(compressed_outputs[i]) - l_min) / span)
    return rewards
