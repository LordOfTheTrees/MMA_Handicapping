"""
Lightweight 1-D Kalman filter for tracking fighter parameter uncertainty.

Design (architecture Section 4.5):
- Uncertainty GROWS with time elapsed since last fight (process noise).
- Uncertainty SHRINKS with each new fight observation (measurement update).
- No aging curves, no assumed mechanics of change — the filter is agnostic
  about WHY a fighter's true skill level might drift. It only models that
  our estimate becomes less certain the longer we go without observing them.
"""
from dataclasses import dataclass


@dataclass
class KalmanState:
    """Current estimate of a scalar parameter with associated uncertainty."""
    value: float      # point estimate
    variance: float   # uncertainty (Kalman variance)


def kalman_predict(
    state: KalmanState,
    days_elapsed: float,
    process_noise_per_day: float,
) -> KalmanState:
    """
    Time update: inflate variance proportional to days elapsed.

    Returns a new KalmanState; does not mutate the input.
    """
    if days_elapsed <= 0:
        return state
    return KalmanState(
        value=state.value,
        variance=state.variance + process_noise_per_day * days_elapsed,
    )


def kalman_update(
    state: KalmanState,
    observation: float,
    measurement_noise: float,
) -> KalmanState:
    """
    Measurement update: incorporate a new observation.

    Uses the standard Kalman gain formula:
        K = P / (P + R)
        x_new = x + K * (z - x)
        P_new = (1 - K) * P

    Returns a new KalmanState; does not mutate the input.
    """
    gain = state.variance / (state.variance + measurement_noise)
    new_value = state.value + gain * (observation - state.value)
    new_variance = (1.0 - gain) * state.variance
    return KalmanState(value=new_value, variance=new_variance)
