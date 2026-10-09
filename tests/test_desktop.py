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
    assert d.APP_VERSION == "0.2.0"
    assert d.APP_NAME == "ModeTrains Desktop"


if __name__ == "__main__":
    for fn in [test_train_command, test_train_validation, test_infer_command,
               test_info_command, test_version]:
        fn(); print(f"PASS {fn.__name__}")
    print("ALL DESKTOP TESTS PASSED")
