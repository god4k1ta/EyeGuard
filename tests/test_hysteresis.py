from src.EyeGuard import hysteresis, HYSTERESIS_UP, HYSTERESIS_DOWN


def test_hysteresis_on():
    assert hysteresis(True, False) is True


def test_hysteresis_hold():
    assert hysteresis(False, True) is True


def test_hysteresis_off():
    assert hysteresis(False, False) is False


def test_hysteresis_threshold():
    assert (HYSTERESIS_UP > HYSTERESIS_DOWN)
