"""Desktop logic tests — no display needed (never instantiates Tk).

Copyright (c) 2026 salim-slimani.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "desktop"))

import modetrains_desktop as d


def test_train_command():
    cmd = d.build_train_command("m", "ds", steps=10, batch=1, accum=2)
    assert cmd[0:3] == [d.find_python(), "-m", "modetrains"]
    assert cmd[3] == "train"
    assert "--steps" in cmd and "10" in cmd
    assert "--batch" in cmd and "1" in cmd


def test_train_validation():
    for bad in [("", "ds"), ("m", ""), ("m", "  ")]:
        try:
            d.build_train_command(*bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"expected ValueError for {bad}")
    try:
        d.build_train_command("m", "d", max_seq=64)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for max_seq<256")


def test_infer_command():
    cmd = d.build_infer_command("m", "hello", tokens=32)
    assert "infer" in cmd and "--tokens" in cmd and "32" in cmd
    assert "--full" not in cmd
    cmd2 = d.build_infer_command("m", "hello", full=True)
    assert "--full" in cmd2
    try:
        d.build_infer_command("m", "   ")
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for empty prompt")


def test_info_command():
    assert d.build_info_command()[-1] == "info"


def test_version():
    assert d.APP_VERSION == "0.2.1"
    assert d.APP_NAME == "ModeTrains Desktop"


def test_frozen_guard():
    # Source run: never refuse.
    assert d.is_frozen() is False
    assert d.should_refuse_args(["app", "-m", "modetrains", "info"]) is False
    # Simulated frozen run.
    sys.frozen = True
    try:
        assert d.is_frozen() is True
        assert d.should_refuse_args(["app"]) is False
        assert d.should_refuse_args(["app", "--self-test"]) is False
        assert d.should_refuse_args(["app", "-m", "modetrains", "info"]) is True
        assert d.should_refuse_args(["app", "-c", "x"]) is True
    finally:
        del sys.frozen


def test_find_backend_no_crash():
    b = d.find_backend_python(timeout=120)
    assert b is None or isinstance(b, str)
    if b is not None and d.is_frozen():
        assert os.path.abspath(b) != os.path.abspath(sys.executable)


if __name__ == "__main__":
    for fn in [test_train_command, test_train_validation, test_infer_command,
               test_info_command, test_version, test_frozen_guard,
               test_find_backend_no_crash]:
        fn(); print(f"PASS {fn.__name__}")
    print("ALL DESKTOP TESTS PASSED")
