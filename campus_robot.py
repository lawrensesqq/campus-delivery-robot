"""
Mini Campus Delivery Robot  —  AI Fundamentals Final Project (Topic 02)

A robot carries a package across a small campus map.
  1. A* search            -> plans the shortest route on a 5x5 grid with walls
  2. Bayesian inference   -> updates P(obstacle) in the narrow corridor after one sensor reading
  3. MDP (Bellman)        -> decides Move or Wait using the probability from step 2

The three parts share one goal: deliver the package safely and quickly.
A* finds that the route must pass a single corridor cell; Bayes estimates
whether that corridor is blocked; the MDP turns that estimate into an action.

All numbers are toy assumptions (no real dataset or trained model).
Run:  python campus_robot.py          (console demo)
      streamlit run app.py            (interactive UI)
      python -m pytest                (tests)
"""

import heapq

# ---------------------------------------------------------------------------
# 0. Campus map (5x5).  (row, col), row 0 is the top.
# ---------------------------------------------------------------------------
ROWS, COLS = 5, 5
START = (0, 0)          # dormitory
GOAL = (4, 4)           # library
WALL_A = [(0, 2), (1, 2), (2, 2)]   # wall 1: vertical building block
WALL_B = [(3, 2), (3, 3), (3, 4)]   # wall 2: horizontal building block
WALLS = set(WALL_A + WALL_B)
CORRIDOR = (4, 2)       # the only gap between the two buildings; has a sensor


def show_grid(path=None, extra_blocked=None):
    path = set(path or [])
    extra_blocked = extra_blocked or set()
    for r in range(ROWS):
        row = []
        for c in range(COLS):
            cell = (r, c)
            if cell == START:
                row.append("S")
            elif cell == GOAL:
                row.append("G")
            elif cell in WALLS:
                row.append("#")
            elif cell in extra_blocked:
                row.append("X")
            elif cell in path:
                row.append("*")
            elif cell == CORRIDOR:
                row.append("C")
            else:
                row.append(".")
        print("   " + " ".join(row))


# ---------------------------------------------------------------------------
# 1. A* search
# ---------------------------------------------------------------------------
def manhattan(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def neighbors(cell, blocked, rows=ROWS, cols=COLS):
    r, c = cell
    for dr, dc in [(1, 0), (0, 1), (-1, 0), (0, -1)]:   # down, right, up, left
        nr, nc = r + dr, c + dc
        if 0 <= nr < rows and 0 <= nc < cols and (nr, nc) not in blocked:
            yield (nr, nc)


def a_star(start, goal, blocked, verbose=True, rows=ROWS, cols=COLS, trace=None):
    """f(n) = g(n) + h(n); g = steps so far, h = Manhattan distance (admissible).

    Returns (path, nodes_expanded); path is None if the goal is unreachable.
    If `trace` is a list, each expansion is appended as (step, cell, g, h, f).
    """
    if start in blocked or goal in blocked:
        return None, 0
    open_heap = [(manhattan(start, goal), 0, start)]   # (f, g, cell)
    came_from = {start: None}
    g_cost = {start: 0}
    closed = set()
    step = 0
    if verbose:
        print("   step | expand  |  g  h  f")
    while open_heap:
        f, g, cell = heapq.heappop(open_heap)
        if cell in closed:
            continue
        closed.add(cell)
        step += 1
        h = manhattan(cell, goal)
        if verbose:
            print(f"   {step:>4} | {str(cell):7} | {g:2} {h:2} {f:2}")
        if trace is not None:
            trace.append((step, cell, g, h, f))
        if cell == goal:
            path = []
            while cell is not None:
                path.append(cell)
                cell = came_from[cell]
            return path[::-1], step
        for nb in neighbors(cell, blocked, rows, cols):
            new_g = g + 1                       # every move costs 1
            if new_g < g_cost.get(nb, float("inf")):
                g_cost[nb] = new_g
                came_from[nb] = cell
                heapq.heappush(open_heap, (new_g + manhattan(nb, goal), new_g, nb))
    return None, step


def bfs(start, goal, blocked, rows=ROWS, cols=COLS):
    """Uninformed baseline (breadth-first search) used to compare against A*.

    Returns (path, nodes_expanded).
    """
    if start in blocked or goal in blocked:
        return None, 0
    frontier = [start]
    came_from = {start: None}
    expanded = 0
    while frontier:
        cell = frontier.pop(0)
        expanded += 1
        if cell == goal:
            path = []
            while cell is not None:
                path.append(cell)
                cell = came_from[cell]
            return path[::-1], expanded
        for nb in neighbors(cell, blocked, rows, cols):
            if nb not in came_from:
                came_from[nb] = cell
                frontier.append(nb)
    return None, expanded


# ---------------------------------------------------------------------------
# 2. Bayesian inference
# ---------------------------------------------------------------------------
PRIOR_OBSTACLE = 0.20          # P(Obstacle): corridor is blocked 20% of the time
P_ALERT_GIVEN_OBSTACLE = 0.90  # sensor sensitivity
P_ALERT_GIVEN_FREE = 0.20      # false-alarm rate (people walking by, etc.)


def bayes_update(prior, observation, p_alert_obs=P_ALERT_GIVEN_OBSTACLE,
                 p_alert_free=P_ALERT_GIVEN_FREE, verbose=True):
    """P(Obs | z) = P(z | Obs) * P(Obs) / P(z)"""
    if not 0.0 <= prior <= 1.0:
        raise ValueError("prior must be in [0, 1]")
    if observation == "alert":
        like_obs, like_free = p_alert_obs, p_alert_free
    elif observation == "no_alert":
        like_obs, like_free = 1 - p_alert_obs, 1 - p_alert_free
    else:
        raise ValueError("observation must be 'alert' or 'no_alert'")
    joint_obs = like_obs * prior
    joint_free = like_free * (1 - prior)
    evidence = joint_obs + joint_free
    if evidence == 0:
        raise ValueError("observation has zero probability under this model")
    posterior = joint_obs / evidence
    if verbose:
        print(f"   observation           : {observation}")
        print(f"   P(z | Obstacle)       = {like_obs:.2f}")
        print(f"   P(z | Free)           = {like_free:.2f}")
        print(f"   P(z | Obs) * P(Obs)   = {like_obs:.2f} * {prior:.2f} = {joint_obs:.4f}")
        print(f"   P(z | Free) * P(Free) = {like_free:.2f} * {1 - prior:.2f} = {joint_free:.4f}")
        print(f"   P(z) (evidence)       = {evidence:.4f}")
        print(f"   P(Obstacle | z)       = {joint_obs:.4f} / {evidence:.4f} = {posterior:.3f}")
    return posterior


def bayes_sequence(prior, observations, **kwargs):
    """Apply several sensor readings one after another (posterior -> next prior).

    Returns the list of beliefs after each reading.
    """
    beliefs = []
    p = prior
    for z in observations:
        p = bayes_update(p, z, verbose=False, **kwargs)
        beliefs.append(p)
    return beliefs


# ---------------------------------------------------------------------------
# 3. MDP: 2 states (Clear, Blocked) x 2 actions (Move, Wait)
# ---------------------------------------------------------------------------
STATES = ["Clear", "Blocked"]
ACTIONS = ["Move", "Wait"]
GAMMA = 0.9

# T[state][action] = {next_state: probability}
T = {
    "Clear":   {"Move": {"Clear": 0.9, "Blocked": 0.1},
                "Wait": {"Clear": 0.9, "Blocked": 0.1}},
    "Blocked": {"Move": {"Clear": 0.2, "Blocked": 0.8},   # bumping in does not clear it
                "Wait": {"Clear": 0.6, "Blocked": 0.4}},  # people tend to move away
}
# R[state][action]
R = {
    "Clear":   {"Move": +10, "Wait": -2},   # progress vs. wasted time
    "Blocked": {"Move": -20, "Wait": -1},   # collision / damaged package vs. small delay
}


def q_value(s, a, V, gamma=GAMMA, rewards=None, transitions=None):
    """Bellman: Q(s,a) = R(s,a) + gamma * sum_s' T(s'|s,a) * V(s')"""
    rewards = rewards or R
    transitions = transitions or T
    return rewards[s][a] + gamma * sum(p * V[s2] for s2, p in transitions[s][a].items())


def value_iteration(tol=1e-4, show_first=True, gamma=GAMMA, rewards=None,
                    transitions=None, verbose=True, max_iter=10_000):
    """Returns (V, Q).  Repeats Bellman updates until V changes by less than tol."""
    if not 0.0 <= gamma < 1.0:
        raise ValueError("gamma must be in [0, 1)")
    V = {s: 0.0 for s in STATES}
    it = 0
    while it < max_iter:
        it += 1
        Q = {s: {a: q_value(s, a, V, gamma, rewards, transitions) for a in ACTIONS}
             for s in STATES}
        newV = {s: max(Q[s].values()) for s in STATES}
        if verbose and show_first and it == 1:
            print("   Iteration 1 (V0 = 0, so Q = immediate reward R):")
            for s in STATES:
                print(f"     Q({s:7},Move) = {Q[s]['Move']:7.2f}   Q({s:7},Wait) = {Q[s]['Wait']:7.2f}")
        if max(abs(newV[s] - V[s]) for s in STATES) < tol:
            if verbose:
                print(f"   Converged after {it} iterations (gamma = {gamma}):")
                for s in STATES:
                    best = max(Q[s], key=Q[s].get)
                    print(f"     V({s:7}) = {newV[s]:7.2f}   Q(Move) = {Q[s]['Move']:7.2f}"
                          f"   Q(Wait) = {Q[s]['Wait']:7.2f}   -> best: {best}")
            value_iteration.iterations = it
            return newV, Q
        V = newV
    raise RuntimeError("value iteration did not converge")


def decide(belief_blocked, Q, verbose=True):
    """The robot is not sure of the state, so weight Q by the Bayes probability."""
    b = belief_blocked
    score = {a: b * Q["Blocked"][a] + (1 - b) * Q["Clear"][a] for a in ACTIONS}
    if verbose:
        for a in ACTIONS:
            print(f"   E[Q({a})] = {b:.3f} * {Q['Blocked'][a]:7.2f} + {1 - b:.3f} * {Q['Clear'][a]:7.2f}"
                  f" = {score[a]:7.2f}")
    action = max(score, key=score.get)
    return action, score


def break_even(Q):
    """Belief b* where Move and Wait are equally good."""
    dm = Q["Clear"]["Move"] - Q["Clear"]["Wait"]
    db = Q["Blocked"]["Move"] - Q["Blocked"]["Wait"]
    return dm / (dm - db)


# ---------------------------------------------------------------------------
# Main demo
# ---------------------------------------------------------------------------
def line(title):
    print("\n" + "=" * 64 + f"\n{title}\n" + "=" * 64)


def run_scenario(observation, Q, path):
    line(f"SCENARIO: corridor sensor reports '{observation}'")
    print("[STEP 2 - Bayesian update]")
    p = bayes_update(PRIOR_OBSTACLE, observation)
    print("\n[STEP 3 - MDP decision with belief P(Blocked) = %.3f]" % p)
    action, _ = decide(p, Q)
    print(f"\n>>> Robot at {path[path.index(CORRIDOR) - 1]} next to corridor {CORRIDOR}: {action.upper()}")
    return p, action


def main():
    line("CAMPUS MAP   S=dorm  G=library  #=wall  C=corridor (sensor)")
    show_grid()

    line("STEP 1 - A* route planning (h = Manhattan distance)")
    path, expanded = a_star(START, GOAL, WALLS)
    print(f"\n   Route ({len(path) - 1} moves, {expanded} nodes expanded):")
    print("   " + " -> ".join(map(str, path)))
    show_grid(path)
    print(f"   Route passes the corridor {CORRIDOR}: {CORRIDOR in path}")

    print("\n   Is there a detour if the corridor is blocked?")
    detour, _ = a_star(START, GOAL, WALLS | {CORRIDOR}, verbose=False)
    print("   A* with corridor removed ->", "no path found" if detour is None else detour)
    print("   => the robot cannot reroute, it must decide: Move through or Wait.")

    line("MDP SETUP - value iteration (Q-values used in step 3)")
    V, Q = value_iteration()
    b_star = break_even(Q)
    print(f"   Break-even belief: Move is better only if P(Blocked) < {b_star:.3f}")

    results = [run_scenario(obs, Q, path) for obs in ["alert", "no_alert"]]

    line("SUMMARY")
    print(f"   Route            : {len(path) - 1} moves via corridor {CORRIDOR}")
    for obs, (p, a) in zip(["alert", "no_alert"], results):
        print(f"   Sensor {obs:9}: P(obstacle) = {p:.3f} -> {a}")


if __name__ == "__main__":
    main()
