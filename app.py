"""Interactive web UI for the Mini Campus Delivery Robot.

Run:  streamlit run app.py
"""

import altair as alt
import pandas as pd
import streamlit as st

import campus_robot as cr

st.set_page_config(page_title="Campus Delivery Robot", page_icon="🤖", layout="wide")

# ---------------------------------------------------------------------------
# State: editable map
# ---------------------------------------------------------------------------
if "walls" not in st.session_state:
    st.session_state.walls = set(cr.WALLS)


def toggle_wall(cell):
    walls = st.session_state.walls
    walls.symmetric_difference_update({cell})


def reset_map():
    st.session_state.walls = set(cr.WALLS)


# ---------------------------------------------------------------------------
# Sidebar: model parameters
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("📡 Corridor sensor")
    n_readings = st.number_input("Number of sensor readings", 1, 5, 1)
    readings = []
    for i in range(int(n_readings)):
        z = st.radio(f"Reading {i + 1}", ["alert", "no_alert"], horizontal=True, key=f"z{i}",
                     format_func=lambda v: "🚨 alert" if v == "alert" else "✅ no alert")
        readings.append(z)

    st.header("🎲 Bayes parameters")
    prior = st.slider("Prior P(obstacle)", 0.0, 1.0, cr.PRIOR_OBSTACLE, 0.01)
    p_alert_obs = st.slider("Sensitivity P(alert | obstacle)", 0.01, 1.0, cr.P_ALERT_GIVEN_OBSTACLE, 0.01)
    p_alert_free = st.slider("False alarm P(alert | free)", 0.0, 0.99, cr.P_ALERT_GIVEN_FREE, 0.01)

    st.header("🧭 MDP parameters")
    gamma = st.slider("Discount γ", 0.0, 0.99, cr.GAMMA, 0.01)
    with st.expander("Rewards"):
        r_cm = st.number_input("Clear, Move", value=float(cr.R["Clear"]["Move"]), step=1.0)
        r_cw = st.number_input("Clear, Wait", value=float(cr.R["Clear"]["Wait"]), step=1.0)
        r_bm = st.number_input("Blocked, Move", value=float(cr.R["Blocked"]["Move"]), step=1.0)
        r_bw = st.number_input("Blocked, Wait", value=float(cr.R["Blocked"]["Wait"]), step=1.0)
    rewards = {"Clear": {"Move": r_cm, "Wait": r_cw}, "Blocked": {"Move": r_bm, "Wait": r_bw}}

    if st.button("↺ Reset all to defaults", width="stretch"):
        st.session_state.clear()
        st.rerun()

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.title("🤖 Mini Campus Delivery Robot")
st.caption("Deliver a package from the dormitory 🏠 to the library 📚 — "
           "**A\\* search** plans the route, **Bayesian inference** reads the noisy corridor sensor, "
           "an **MDP** decides whether to move or wait.")

walls = st.session_state.walls
trace = []
path, expanded = cr.a_star(cr.START, cr.GOAL, walls, verbose=False, trace=trace)
_, bfs_expanded = cr.bfs(cr.START, cr.GOAL, walls)
uses_corridor = path is not None and cr.CORRIDOR in path
detour, _ = cr.a_star(cr.START, cr.GOAL, walls | {cr.CORRIDOR}, verbose=False)

try:
    beliefs = cr.bayes_sequence(prior, readings, p_alert_obs=p_alert_obs, p_alert_free=p_alert_free)
except ValueError:
    st.error("These sensor readings are impossible under the chosen parameters "
             "(probability of the observation is 0). Change the prior or the sensor model.")
    st.stop()
belief = beliefs[-1]
V, Q = cr.value_iteration(gamma=gamma, rewards=rewards, verbose=False)
vi_iterations = cr.value_iteration.iterations
action, score = cr.decide(belief, Q, verbose=False)
try:
    b_star = cr.break_even(Q)
except ZeroDivisionError:   # Move and Wait differ by the same amount in both states
    b_star = None

# ---------------------------------------------------------------------------
# Final decision banner
# ---------------------------------------------------------------------------
if path is None:
    st.error("🚫 No route exists from the dorm to the library on this map — remove some walls.")
elif not uses_corridor:
    st.success(f"✅ The shortest route ({len(path) - 1} moves) avoids the corridor, "
               "so the sensor does not matter — the robot just drives.")
elif detour is not None and len(detour) == len(path):
    st.success("✅ An equally short detour exists, so the robot can avoid the corridor entirely.")
else:
    icon = "🟢 MOVE through the corridor" if action == "Move" else "🟡 WAIT before the corridor"
    st.info(f"### Decision: {icon}\n"
            f"P(corridor blocked) = **{belief:.3f}** · "
            f"E[Q(Move)] = {score['Move']:.2f} vs E[Q(Wait)] = {score['Wait']:.2f}"
            + (f" · break-even = {b_star:.3f}" if b_star is not None and 0 <= b_star <= 1 else ""))

tab1, tab2, tab3 = st.tabs(["1️⃣ A* route planning", "2️⃣ Bayesian sensor update", "3️⃣ MDP decision"])

# ---------------------------------------------------------------------------
# Tab 1: A*
# ---------------------------------------------------------------------------
COLORS = {"wall": "#475569", "path": "#bfdbfe", "corridor": "#fde68a", "free": "#f8fafc"}


def cell_html(cell):
    on_path = path is not None and cell in path
    if cell == cr.START:
        label, bg = "🏠", COLORS["path"]
    elif cell == cr.GOAL:
        label, bg = "📚", COLORS["path"]
    elif cell in walls:
        label, bg = "", COLORS["wall"]
    elif cell == cr.CORRIDOR:
        label = "🚧" if belief >= 0.5 else "📡"
        bg = COLORS["corridor"]
    elif on_path:
        label, bg = "•", COLORS["path"]
    else:
        label, bg = "", COLORS["free"]
    if uses_corridor and path and cell == path[path.index(cr.CORRIDOR) - 1]:
        label = "🤖"
    return (f'<td style="width:64px;height:64px;background:{bg};text-align:center;'
            f'font-size:28px;border:2px solid #e2e8f0;color:#1e3a8a">{label}</td>')


with tab1:
    left, right = st.columns([1, 1])
    with left:
        st.subheader("Campus map")
        html = "<table style='border-collapse:collapse'>"
        for r in range(cr.ROWS):
            html += "<tr>" + "".join(cell_html((r, c)) for c in range(cr.COLS)) + "</tr>"
        html += "</table>"
        st.markdown(html, unsafe_allow_html=True)
        st.caption("🏠 dorm · 📚 library · 🤖 robot waiting point · 📡 corridor sensor "
                   "(🚧 if P(blocked) ≥ 0.5) · dark = building · blue = A* route")

        with st.expander("✏️ Edit the map (click a cell to add/remove a wall)"):
            for r in range(cr.ROWS):
                cols = st.columns(cr.COLS)
                for c in range(cr.COLS):
                    cell = (r, c)
                    if cell in (cr.START, cr.GOAL, cr.CORRIDOR):
                        cols[c].button({cr.START: "🏠", cr.GOAL: "📚", cr.CORRIDOR: "📡"}[cell],
                                       key=f"b{r}{c}", disabled=True, width="stretch")
                    else:
                        cols[c].button("⬛" if cell in walls else "⬜", key=f"b{r}{c}",
                                       on_click=toggle_wall, args=(cell,), width="stretch")
            st.button("Reset map", on_click=reset_map)

    with right:
        st.subheader("Result")
        m1, m2, m3 = st.columns(3)
        m1.metric("Route length", "—" if path is None else f"{len(path) - 1} moves")
        m2.metric("A* nodes expanded", expanded)
        m3.metric("BFS nodes expanded", bfs_expanded,
                  delta=f"{bfs_expanded - expanded} more than A*", delta_color="off")
        if path:
            st.code(" → ".join(map(str, path)), language=None)
        st.write(f"Route goes through the corridor {cr.CORRIDOR}: **{uses_corridor}**  \n"
                 f"Detour if the corridor is blocked: **{'none' if detour is None else f'{len(detour) - 1} moves'}**")
        st.markdown("**Expansion order** — A\\* always expands the node with the smallest "
                    "f = g + h (h = Manhattan distance, admissible ⇒ optimal route).")
        st.dataframe(pd.DataFrame([{"step": s, "cell": str(c), "g": g, "h": h, "f": f}
                                   for s, c, g, h, f in trace]),
                     hide_index=True, width="stretch", height=300)

# ---------------------------------------------------------------------------
# Tab 2: Bayes
# ---------------------------------------------------------------------------
with tab2:
    st.subheader("How sure are we that the corridor is blocked?")
    st.latex(r"P(\text{Obs}\mid z)=\frac{P(z\mid \text{Obs})\,P(\text{Obs})}"
             r"{P(z\mid \text{Obs})P(\text{Obs})+P(z\mid \text{Free})P(\text{Free})}")
    rows = []
    p = prior
    for i, z in enumerate(readings, 1):
        lo, lf = (p_alert_obs, p_alert_free) if z == "alert" else (1 - p_alert_obs, 1 - p_alert_free)
        post = beliefs[i - 1]
        rows.append({"reading": i, "observation": z, "prior": round(p, 4),
                     "P(z|Obs)": round(lo, 2), "P(z|Free)": round(lf, 2),
                     "P(z|Obs)·P(Obs)": round(lo * p, 4),
                     "evidence P(z)": round(lo * p + lf * (1 - p), 4),
                     "posterior": round(post, 4)})
        p = post
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")

    chart_df = pd.DataFrame({"stage": ["prior"] + [f"after reading {i}" for i in range(1, len(beliefs) + 1)],
                             "P(obstacle)": [prior] + beliefs})
    chart = alt.Chart(chart_df).mark_bar(color="#3b82f6").encode(
        x=alt.X("stage:N", sort=None, title=None),
        y=alt.Y("P(obstacle):Q", scale=alt.Scale(domain=[0, 1])),
        tooltip=["stage", alt.Tooltip("P(obstacle):Q", format=".3f")])
    st.altair_chart(chart, width="stretch")
    st.caption("The sensor is noisy: one alert raises the probability but does not prove the corridor is blocked. "
               "Several readings are combined by using each posterior as the next prior.")

# ---------------------------------------------------------------------------
# Tab 3: MDP
# ---------------------------------------------------------------------------
with tab3:
    st.subheader("Move or wait? — value iteration (Bellman)")
    st.latex(r"Q(s,a)=R(s,a)+\gamma\sum_{s'}T(s'\mid s,a)\,V(s'),\qquad V(s)=\max_a Q(s,a)")
    c1, c2 = st.columns([1, 1])
    with c1:
        st.markdown(f"Converged after **{vi_iterations}** iterations (γ = {gamma})")
        st.dataframe(pd.DataFrame([{"state": s, "Q(Move)": round(Q[s]["Move"], 2),
                                    "Q(Wait)": round(Q[s]["Wait"], 2), "V(s)": round(V[s], 2),
                                    "best": max(Q[s], key=Q[s].get)} for s in cr.STATES]),
                     hide_index=True, width="stretch")
        st.markdown("The robot does not know the true state, so it weights Q by the belief from Bayes:")
        st.latex(rf"E[Q(a)] = {belief:.3f}\cdot Q(\text{{Blocked}},a) + {1 - belief:.3f}\cdot Q(\text{{Clear}},a)")
        st.write(f"E[Q(Move)] = **{score['Move']:.2f}**, E[Q(Wait)] = **{score['Wait']:.2f}** → **{action.upper()}**")
        if b_star is not None and 0 <= b_star <= 1:
            st.write(f"Break-even belief: Move is better only while P(blocked) < **{b_star:.3f}**")

    with c2:
        bs = [i / 100 for i in range(101)]
        lines = pd.DataFrame([{"P(blocked)": b, "action": a,
                               "expected_Q": b * Q["Blocked"][a] + (1 - b) * Q["Clear"][a]}
                              for b in bs for a in cr.ACTIONS])
        base = alt.Chart(lines).mark_line(strokeWidth=3).encode(
            x="P(blocked):Q",
            y=alt.Y("expected_Q:Q", title="Expected Q", scale=alt.Scale(zero=False)),
            color=alt.Color("action:N", scale=alt.Scale(domain=["Move", "Wait"],
                                                        range=["#16a34a", "#f59e0b"])))
        rule = alt.Chart(pd.DataFrame({"x": [belief]})).mark_rule(color="#dc2626", strokeDash=[6, 4]) \
            .encode(x="x:Q")
        st.altair_chart(base + rule, width="stretch")
        st.caption("Expected value of each action vs. belief. Red dashed line = current belief.")
