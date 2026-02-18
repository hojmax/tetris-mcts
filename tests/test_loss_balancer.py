import pytest

from tetris_bot.ml.loss import RunningLossBalancer


def test_running_loss_balancer_requires_positive_window() -> None:
    with pytest.raises(ValueError, match="window_size must be > 0"):
        RunningLossBalancer(0)


def test_running_loss_balancer_requires_history() -> None:
    balancer = RunningLossBalancer(3)

    with pytest.raises(ValueError, match="without history"):
        balancer.averages()

    with pytest.raises(ValueError, match="without history"):
        balancer.value_loss_weight()


def test_running_loss_balancer_tracks_ratio_from_running_averages() -> None:
    balancer = RunningLossBalancer(3)
    balancer.append(policy_loss=4.0, value_loss=2.0)
    balancer.append(policy_loss=10.0, value_loss=5.0)
    balancer.append(policy_loss=8.0, value_loss=4.0)

    policy_avg, value_avg = balancer.averages()
    assert policy_avg == pytest.approx((4.0 + 10.0 + 8.0) / 3.0)
    assert value_avg == pytest.approx((2.0 + 5.0 + 4.0) / 3.0)
    assert balancer.value_loss_weight() == pytest.approx(policy_avg / value_avg)


def test_running_loss_balancer_uses_last_n_losses() -> None:
    balancer = RunningLossBalancer(2)
    balancer.append(policy_loss=6.0, value_loss=3.0)
    balancer.append(policy_loss=10.0, value_loss=2.0)
    balancer.append(policy_loss=12.0, value_loss=6.0)

    policy_avg, value_avg = balancer.averages()
    assert policy_avg == pytest.approx((10.0 + 12.0) / 2.0)
    assert value_avg == pytest.approx((2.0 + 6.0) / 2.0)
    assert balancer.value_loss_weight() == pytest.approx(policy_avg / value_avg)


def test_running_loss_balancer_rejects_invalid_losses() -> None:
    balancer = RunningLossBalancer(2)

    with pytest.raises(ValueError, match="policy_loss must be >= 0"):
        balancer.append(policy_loss=-1.0, value_loss=1.0)

    with pytest.raises(ValueError, match="value_loss must be > 0"):
        balancer.append(policy_loss=1.0, value_loss=0.0)
