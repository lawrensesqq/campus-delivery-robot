"""Unit tests for the three AI components and how they work together.

Run:  python -m pytest -v
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import campus_robot as cr  # noqa: E402


# ---------------------------------------------------------------------------
# 1. A* search
# ---------------------------------------------------------------------------
class TestAStar:
    def test_finds_shortest_route_on_campus_map(self):
        path, expanded = cr.a_star(cr.START, cr.GOAL, cr.WALLS, verbose=False)
        assert path[0] == cr.START and path[-1] == cr.GOAL
        assert len(path) - 1 == 8
        assert expanded == 13

    def test_route_is_valid(self):
        path, _ = cr.a_star(cr.START, cr.GOAL, cr.WALLS, verbose=False)
        for a, b in zip(path, path[1:]):
            assert cr.manhattan(a, b) == 1          # one step at a time
        assert not set(path) & cr.WALLS               # never walks through a wall

    def test_route_uses_corridor(self):
        path, _ = cr.a_star(cr.START, cr.GOAL, cr.WALLS, verbose=False)
        assert cr.CORRIDOR in path

    def test_no_path_when_corridor_blocked(self):
        path, _ = cr.a_star(cr.START, cr.GOAL, cr.WALLS | {cr.CORRIDOR}, verbose=False)
        assert path is None

    def test_matches_bfs_length(self):
        """A* with an admissible heuristic must be as short as BFS (optimal)."""
        a_path, _ = cr.a_star(cr.START, cr.GOAL, cr.WALLS, verbose=False)
        b_path, _ = cr.bfs(cr.START, cr.GOAL, cr.WALLS)
        assert len(a_path) == len(b_path)

    def test_expands_no_more_than_bfs(self):
        _, a_exp = cr.a_star(cr.START, cr.GOAL, cr.WALLS, verbose=False)
        _, b_exp = cr.bfs(cr.START, cr.GOAL, cr.WALLS)
        assert a_exp <= b_exp

    def test_empty_grid_straight_line(self):
        path, _ = cr.a_star((0, 0), (0, 4), set(), verbose=False)
        assert path == [(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)]

    def test_start_equals_goal(self):
        path, expanded = cr.a_star((2, 2), (2, 2), set(), verbose=False)
        assert path == [(2, 2)] and expanded == 1

    def test_blocked_start_or_goal(self):
        assert cr.a_star((0, 0), (4, 4), {(0, 0)}, verbose=False)[0] is None
        assert cr.a_star((0, 0), (4, 4), {(4, 4)}, verbose=False)[0] is None

    def test_heuristic_is_admissible(self):
        """h(n) <= true remaining distance for every reachable cell."""
        for r in range(cr.ROWS):
            for c in range(cr.COLS):
                cell = (r, c)
                if cell in cr.WALLS:
                    continue
                path, _ = cr.bfs(cell, cr.GOAL, cr.WALLS)
                if path is not None:
                    assert cr.manhattan(cell, cr.GOAL) <= len(path) - 1

    def test_trace_records_expansions(self):
        trace = []
        _, expanded = cr.a_star(cr.START, cr.GOAL, cr.WALLS, verbose=False, trace=trace)
        assert len(trace) == expanded
        assert trace[0][1] == cr.START and trace[-1][1] == cr.GOAL
        for _, _, g, h, f in trace:
            assert f == g + h


# ---------------------------------------------------------------------------
# 2. Bayesian inference
# ---------------------------------------------------------------------------
class TestBayes:
    def test_alert_matches_hand_calculation(self):
        p = cr.bayes_update(0.2, "alert", verbose=False)
        assert p == pytest.approx(0.18 / 0.34)
        assert round(p, 3) == 0.529

    def test_no_alert_matches_hand_calculation(self):
        p = cr.bayes_update(0.2, "no_alert", verbose=False)
        assert p == pytest.approx(0.02 / 0.66)
        assert round(p, 3) == 0.030

    def test_alert_raises_and_silence_lowers_belief(self):
        assert cr.bayes_update(0.2, "alert", verbose=False) > 0.2
        assert cr.bayes_update(0.2, "no_alert", verbose=False) < 0.2

    def test_useless_sensor_does_not_change_belief(self):
        p = cr.bayes_update(0.2, "alert", p_alert_obs=0.5, p_alert_free=0.5, verbose=False)
        assert p == pytest.approx(0.2)

    def test_extreme_priors(self):
        assert cr.bayes_update(0.0, "alert", verbose=False) == 0.0
        assert cr.bayes_update(1.0, "no_alert", verbose=False) == 1.0

    def test_posteriors_are_probabilities(self):
        for prior in [0.01, 0.2, 0.5, 0.9]:
            for z in ["alert", "no_alert"]:
                assert 0.0 <= cr.bayes_update(prior, z, verbose=False) <= 1.0

    def test_invalid_input(self):
        with pytest.raises(ValueError):
            cr.bayes_update(0.2, "beep", verbose=False)
        with pytest.raises(ValueError):
            cr.bayes_update(1.5, "alert", verbose=False)

    def test_two_alerts_increase_confidence(self):
        beliefs = cr.bayes_sequence(0.2, ["alert", "alert"])
        assert beliefs[1] > beliefs[0] > 0.2
        assert beliefs[1] == pytest.approx(0.81 * 0.2 / (0.81 * 0.2 + 0.04 * 0.8))


# ---------------------------------------------------------------------------
# 3. MDP / value iteration
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def solved():
    return cr.value_iteration(verbose=False)


class TestMDP:
    def test_transition_rows_sum_to_one(self):
        for s in cr.STATES:
            for a in cr.ACTIONS:
                assert sum(cr.T[s][a].values()) == pytest.approx(1.0)

    def test_first_iteration_equals_rewards(self):
        V0 = {s: 0.0 for s in cr.STATES}
        for s in cr.STATES:
            for a in cr.ACTIONS:
                assert cr.q_value(s, a, V0) == cr.R[s][a]

    def test_converged_values(self, solved):
        V, Q = solved
        assert V["Clear"] == pytest.approx(86.44, abs=0.01)
        assert V["Blocked"] == pytest.approx(71.37, abs=0.01)
        assert Q["Blocked"]["Move"] == pytest.approx(46.94, abs=0.01)
        assert Q["Clear"]["Wait"] == pytest.approx(74.44, abs=0.01)

    def test_values_satisfy_bellman_equation(self, solved):
        V, _ = solved
        for s in cr.STATES:
            best = max(cr.q_value(s, a, V) for a in cr.ACTIONS)
            assert best == pytest.approx(V[s], abs=1e-3)

    def test_optimal_policy(self, solved):
        _, Q = solved
        assert max(Q["Clear"], key=Q["Clear"].get) == "Move"
        assert max(Q["Blocked"], key=Q["Blocked"].get) == "Wait"

    def test_break_even(self, solved):
        _, Q = solved
        b = cr.break_even(Q)
        assert b == pytest.approx(0.329, abs=1e-3)
        score = {a: b * Q["Blocked"][a] + (1 - b) * Q["Clear"][a] for a in cr.ACTIONS}
        assert score["Move"] == pytest.approx(score["Wait"])

    def test_invalid_gamma(self):
        with pytest.raises(ValueError):
            cr.value_iteration(gamma=1.0, verbose=False)

    def test_myopic_robot(self):
        """With gamma = 0 the robot only looks at immediate reward, so Q == R."""
        _, Q = cr.value_iteration(gamma=0.0, verbose=False)
        assert Q == {s: {a: float(cr.R[s][a]) for a in cr.ACTIONS} for s in cr.STATES}


# ---------------------------------------------------------------------------
# Integration: A* -> Bayes -> MDP
# ---------------------------------------------------------------------------
class TestPipeline:
    def test_alert_leads_to_wait(self, solved):
        _, Q = solved
        p = cr.bayes_update(cr.PRIOR_OBSTACLE, "alert", verbose=False)
        action, score = cr.decide(p, Q, verbose=False)
        assert action == "Wait"
        assert score["Move"] == pytest.approx(65.53, abs=0.01)
        assert score["Wait"] == pytest.approx(72.81, abs=0.01)

    def test_no_alert_leads_to_move(self, solved):
        _, Q = solved
        p = cr.bayes_update(cr.PRIOR_OBSTACLE, "no_alert", verbose=False)
        action, score = cr.decide(p, Q, verbose=False)
        assert action == "Move"
        assert score["Move"] == pytest.approx(85.24, abs=0.01)

    def test_decision_flips_at_break_even(self, solved):
        _, Q = solved
        b = cr.break_even(Q)
        assert cr.decide(b - 0.01, Q, verbose=False)[0] == "Move"
        assert cr.decide(b + 0.01, Q, verbose=False)[0] == "Wait"

    def test_console_demo_runs(self, capsys):
        cr.main()
        out = capsys.readouterr().out
        assert "Sensor alert    : P(obstacle) = 0.529 -> Wait" in out
        assert "Sensor no_alert : P(obstacle) = 0.030 -> Move" in out
