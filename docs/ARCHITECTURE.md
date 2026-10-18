# Architecture

## Process layout

```
run.sh ──> python -m tourism.cli <command>
              │
              ├── Workspace (one per process): config, City, EngineClient, TravelMatrix, forecaster, policies
              │       │
              │       └── build/tourism_engine   (C++ subprocess, JSON lines on stdin/stdout)
              │
              └── evaluate: ProcessPoolExecutor, one Workspace + engine per worker
```

## C++ engine (`cpp/`)

| File | Responsibility |
| --- | --- |
| `json.hpp/.cpp` | Minimal JSON value, parser and serializer (no dependencies) |
| `city.hpp/.cpp` | Nodes, multimodal edges, transit lines, POIs; day `Conditions` (closures, capacity factors, line delays and suspensions, road closures, congestion); `Context` (forecast tables + planned load); `estimateCrowd` |
| `search.hpp/.cpp` | `Router`: UCS, Greedy best-first and A* over states (node, riding mode/line). The cost is time + money + CO₂ + distance + transfers + transit crowding, all time-dependent. The heuristic is the weighted straight-line distance divided by the fastest allowed speed (admissible and consistent because every edge is at least as long as the straight line). `routeToMany` runs one-to-many uniform-cost search |
| `csp.hpp/.cpp` | `LegRouter`: a leg cache keyed by (from, to, time regime, modes), filled by one-to-many searches. `CspSolver`: domains are start times per candidate (or SKIP); unary filtering; AC-3 on must-visit pairs; a greedy incumbent; DFS branch & bound with MRV, degree and LCV ordering and forward checking. Unsolvable requests get conflict analysis; soft constraints that were made hard are relaxed in order |
| `planner.hpp/.cpp` | `Planner`: the action model and A* over states (location, time, budget, energy, hunger, walked, done / skipped / shifted, lunch). `follow()` executes an agenda; `replan()` tries keep, then min-conflicts repair (shift / move / substitute / skip), then full A* |
| `engine.hpp/.cpp` | Request dispatch: `load_city`, `set_context`, `add_load`, `route`, `matrix`, `bench`, `solve`, `plan`, `replan`, `city_info` |
| `main.cpp` | JSON-lines server (or a single request with `--request file.json`) |

### Hard and soft constraints in the CSP

| Hard (never violated) | Soft (penalised, reported when violated) |
| --- | --- |
| Opening hours, incl. the expected queue | Crowd above the traveller's threshold |
| No overlap, incl. travel between stops | Expected waiting time |
| Day window and return to the end point | CO₂ above the target |
| Budget (entries + fares) | Transfers above the maximum |
| Maximum walking distance | |
| Wheelchair accessibility | |
| Attraction capacity (expected load ≤ 1.6× capacity) | |
| Closures | |
| Must-visits | |
| Lunch inside the lunch window | |
| Pace (maximum number of activities) | |

Soft constraints can be turned into hard ones (`soft_as_hard_crowd`, `soft_as_hard_co2`). The solver then relaxes them in `relax_order` when nothing is feasible.

## Python package (`tourism/`)

| Package | Responsibility |
| --- | --- |
| `core` | `TravellerProfile`, `ReasonRecord`, `Proposal`, `PlanResult`; config loading; `City` |
| `data` | Synthetic city generator; crowd history from the truth model; traveller personas with hidden traits |
| `engine` | `EngineClient` |
| `m1_search` | Utility (interest match × quality), candidate selection, similar alternatives, routing wrappers |
| `m2_planning` | Action table and the PDDL domain / problem export |
| `m3_uncertainty` | `BayesianNetwork`, `CrowdForecaster` (demand multiplier over a per-POI baseline), `OnlineCrowdUpdater`, `plan_risk` |
| `m4_rl` | `StopOptions` and state features (shared by training and the live pipeline), `ItineraryEnv`, agents, training |
| `coordinator` | `base.py` (the three seams), `RuleCoordinator` + `DayState`, `validate_plan`, `TemplateExplainer` |
| `sim` | `TruthModel` (hidden demand and delays), scenarios, `SimulatedTraveller`, `DaySimulator` |
| `eval` | `Workspace`, experiment runner, HTML report |
| `genai` | `ToolBox` (working tools) and agent stubs |

## Data flow for one traveller

1. `DayState` holds the strategy's forecast (Bayesian, historical or none) and the live conditions, and mirrors the planned load in the engine context.
2. `select_candidates` picks the top activities by utility, the best local businesses, three restaurants and the must-visits.
3. The engine's `solve` returns activities, legs, totals, soft violations, conflicts and solver statistics.
4. For each stop, `build_stop_options` computes the alternatives (quieter slot, similar POI, greener leg, nearby local business), and the policy picks an action. Proposals go to the `TravellerInterface`; accepted changes become CSP modifications (forbidden windows, exclusions, extra must-visits, per-leg modes) and the CSP is re-solved.
5. The engine's `plan` returns timed actions, which `validate_plan` checks.
6. `plan_risk` samples crowd multipliers and delays; when a plan is likely to fail, a plan B without the riskiest stop is kept.
7. The explainer phrases the `ReasonRecord`s, and the plan's visits are added to the day's planned load.

## Simulation

Each 15-minute slot runs these steps in order: timeline events, sensor observations (Gamma–Poisson update and surge detection), travellers advancing (arrivals register in queues), queue update (`Q ← max(0, Q + A − c)`, wait = backlog / capacity), arrival resolution (visit or failure). Systems with replanning re-plan an affected traveller at their next step.

Background crowds, sensor noise and travellers are seeded per run, so all systems see the same day.
