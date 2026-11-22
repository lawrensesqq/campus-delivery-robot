"""UI tests for the Streamlit app (headless, using streamlit.testing.AppTest)."""

import os

from streamlit.testing.v1 import AppTest

APP = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "app.py")


def run_app():
    at = AppTest.from_file(APP, default_timeout=30)
    at.run()
    assert not at.exception
    return at


def banner(at):
    return " ".join(m.value for m in at.info) + " ".join(m.value for m in at.success) + \
        " ".join(m.value for m in at.error)


def test_default_alert_waits():
    at = run_app()
    text = banner(at)
    assert "WAIT" in text and "0.529" in text


def test_no_alert_moves():
    at = run_app()
    at.radio(key="z0").set_value("no_alert").run()
    assert not at.exception
    text = banner(at)
    assert "MOVE" in text and "0.030" in text


def test_two_alerts_raise_belief():
    at = run_app()
    at.number_input[0].set_value(2).run()
    at.radio(key="z1").set_value("alert").run()
    assert not at.exception
    assert "0.835" in banner(at)        # 0.162 / 0.194 after two alerts


def test_blocking_the_only_exit_shows_no_route():
    at = run_app()
    at.button(key="b43").click().run()  # wall next to the library
    assert not at.exception
    assert "No route" in banner(at)


def test_one_gap_is_not_enough():
    at = run_app()
    at.button(key="b12").click().run()  # gap in the vertical building only
    assert not at.exception
    assert "WAIT" in banner(at)         # second building still forces the corridor


def test_removing_two_walls_opens_a_detour():
    at = run_app()
    at.button(key="b12").click().run()  # gap in the vertical building
    at.button(key="b34").click().run()  # gap in the horizontal building
    assert not at.exception
    text = banner(at)
    assert "avoids the corridor" in text or "detour" in text
