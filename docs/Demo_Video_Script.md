# Demo Video Script (2–3 minutes)

**Before recording:** run `streamlit run app.py`, open http://localhost:8501, click "Reset all to defaults",
and start a screen recorder (Windows: `Win + Alt + R` in Xbox Game Bar, or OBS). Zoom the browser to about 110%.

| Time | On screen | Say (English) |
|---|---|---|
| 0:00–0:20 | Title of the app, decision banner | "Hi, this is our AI Fundamentals final project, the Mini Campus Delivery Robot. A robot delivers a package from the dormitory to the library. It has to find a route, deal with a noisy sensor, and decide whether to move or wait. We combine three AI topics: A\* search, Bayesian inference and a Markov decision process." |
| 0:20–0:50 | Tab 1: map, route, metrics, expansion table | "First, A\* plans the route. It always expands the cell with the smallest f = g + h. h is the Manhattan distance, which never overestimates, so the route is optimal: 8 moves. The route must pass this corridor, and there is no detour." |
| 0:50–1:05 | Open "Edit the map", click the cell left of the library (4,3) → "No route", then Reset map | "If we put a building here, A\* correctly reports that no route exists. Let's reset the map." |
| 1:05–1:35 | Tab 2: Bayes table and chart | "The corridor has a noisy sensor. It detects obstacles 90% of the time but gives false alarms 20% of the time. After an alert, Bayes' rule gives 0.18 divided by 0.34, which is 53%. That's higher than the 20% prior, but far from certain." |
| 1:35–2:10 | Tab 3: Q-table and line chart | "Value iteration solves the MDP with the Bellman equation and converges after 109 iterations. If the corridor is clear, moving is best; if it's blocked, waiting is best. The robot weights these Q-values by the 53% belief: Wait scores 72.8 and Move scores 65.5, so the robot waits. The lines cross at 0.329, which is the break-even point." |
| 2:10–2:30 | Sidebar: switch Reading 1 to "no alert" → banner turns green MOVE. Then set 2 readings, both alerts → 0.835 | "With no alert, the probability drops to 3% and the robot moves. With two alerts, the probability rises to 84%." |
| 2:30–2:50 | Terminal: `python -m pytest` → 37 passed | "37 automated tests check each algorithm, the full pipeline and the UI. Thanks for watching!" |

**Tips:** keep it under 3 minutes, speak slowly, and keep the mouse still while you talk.
