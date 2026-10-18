# Sustainable and Crowd-Aware Tourism Planner

AI course project (Group 15). The planner recommends enjoyable day itineraries while managing crowds, queues, carbon emissions, attraction capacity and support for local businesses. It balances each traveller's satisfaction against the collective effect of sending many travellers to the same places.

Modules 1–4 are fully implemented and run end to end. Module 5 (GenAI) is prepared as hooks and is added later (see [docs/GENAI_INTEGRATION.md](docs/GENAI_INTEGRATION.md)).

| Module | What it does | Where |
| --- | --- | --- |
| 1 Search + CSP | UCS, Greedy best-first and A* on a time-dependent multimodal graph (walk, metro/tram/bus, taxi). An itinerary CSP with hard and soft constraints, solved by greedy + branch & bound with MRV, degree, LCV, forward checking and AC-3 | `cpp/src/search.cpp`, `cpp/src/csp.cpp` |
| 2 Planning | Actions with preconditions, effects, duration and resources (travel, ticket, visit, eat, local experience, shift to off-peak, rest, skip). Forward A* planner, then replanning in three steps: keep, min-conflicts local repair, full A* replan. PDDL export | `cpp/src/planner.cpp`, `tourism/m2_planning/` |
| 3 Uncertainty | Bayesian networks for crowd level, waiting time and transport delay, learned from history (hierarchical Dirichlet smoothing, variable elimination, soft evidence). Online Gamma–Poisson updating from live sensors. Monte Carlo plan risk with a plan B | `tourism/m3_uncertainty/` |
| 4 RL | A per-stop MDP: keep, shift, substitute, switch mode, add a local business, or ask. Q-learning, SARSA and a LinUCB bandit, compared with satisfaction-only, minimum-distance, environment-only, rule-based and random policies | `tourism/m4_rl/` |
| 5 GenAI | Hooks only: a working tool registry and stub agents | `tourism/genai/` |

## Run it anywhere: exact steps

**1. Install the two prerequisites**

| OS | Command |
| --- | --- |
| macOS | `xcode-select --install` (provides `clang++` and `python3`) |
| Ubuntu / Debian | `sudo apt install -y g++ python3` |
| Fedora | `sudo dnf install -y gcc-c++ python3` |
| Windows | Install WSL (`wsl --install`), then follow the Ubuntu steps inside WSL |

You need Python 3.8 or newer and a C++17 compiler. No pip packages, no internet access and no GPU are needed.

**2. Get the project onto the machine and open a terminal in it**

```bash
git clone https://github.com/jeeban-jyoti/AI-Project-7th-Sem.git
cd AI-Project-7th-Sem            # or: cd "/path/to/AI PROJECT" if you copied the folder (quotes matter with spaces)
chmod +x run.sh                  # once, if the file lost its executable bit when copied
```

**3. Run everything**

```bash
./run.sh all
```

This checks the toolchain, compiles the C++ engine, generates the city and its history, runs the C++ and Python tests, trains the models, runs the benchmarks and the demo, simulates every scenario, and writes the report. It takes about 10–20 minutes on a laptop; the evaluation step uses several CPU cores.

**4. Open the report**

The folder already contains the results of one complete run (`data/` and `outputs/`), so the report can be opened before running anything. `./run.sh all` regenerates all of it.

```bash
open outputs/report/report.html          # macOS
xdg-open outputs/report/report.html      # Linux
```

To run the full evaluation (5 seeds, 200 travellers), use `MODE=full ./run.sh all`, or `./run.sh evaluate --mode full` followed by `./run.sh report --mode full`.

## Everyday commands

```bash
./run.sh demo                                             # conventional vs sustainable plan, an impossible request, an event day
./run.sh plan --profile examples/profiles/family_wheelchair.json
./run.sh plan --interactive                               # fill a short form, accept or reject suggestions yourself
./run.sh plan --profile examples/profiles/foodie_shopper.json --system conventional --scenario S6_rain
./run.sh simulate --scenario S3_public_event --system sustainable --travellers 150
./run.sh bench                                            # UCS vs Greedy vs A*, CSP statistics
./run.sh tools                                            # the GenAI tool registry (already working)
./run.sh pddl                                             # planning domain + PDDL export
./run.sh test                                             # C++ and Python unit tests
./run.sh help
```

Steps can also be run one at a time: `setup`, `build`, `data`, `train`, `evaluate`, `report`. A step that needs an earlier one (for example `demo` needs a built engine and trained models) runs it automatically.

## How the whole system works

```
traveller profile (JSON / form)                                      <- GenAI traveller agent later
        |
        v
M3  Bayesian forecast for the day (calendar, season, weather / rain probability, announced events)
    + load of travellers already planned (collective view)
        |
        v
M1  candidates (interest x quality) -> CSP (A* travel legs inside) -> feasible itinerary
        |                                    unsolvable? -> conflict explained, must-visits relaxed
        v
M4  learned policy, per stop: keep / shift / substitute / switch mode / add local / ask
        |  each suggestion goes to the traveller (simulated now)      <- GenAI traveller agent later
        |  accepted changes -> CSP re-solve (reverted one by one if infeasible)
        v
M2  planner: timed actions with energy, hunger, budget, walking; off-peak waits
        |
        v
validator (Python, independent of the engine) -> M3 risk -> plan B if P(fail) > 30%
        |
        v
explainer (templates over ReasonRecords)                              <- GenAI coordinator later
        |
        v
city simulator: queues, delays, closures, live sensors -> online update -> M2 replanning
```

The C++ engine is one long-lived process per Python worker that speaks JSON over stdin/stdout (`route`, `matrix`, `bench`, `solve`, `plan`, `replan`, `set_context`, `add_load`). [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) has the details.

## Evaluation

Systems: conventional (satisfaction only), minimum distance only, environmental impact only, and the full sustainable planner. Ablations remove the Bayesian forecast, the learned policy, or replanning.

Scenarios (`tourism/sim/scenarios.py`):

| Code | Scenario |
| --- | --- |
| S0 | Normal shoulder-season weekday |
| S1 | Peak-season weekend |
| S2 | Sudden crowd formation (unannounced; detected by live sensors) |
| S3 | Major public event in the Old Town (closures, roads closed) |
| S4 | Transport delays and a line suspension |
| S5 | Attraction capacity limits (timed entry) |
| S6 | Rain (60% forecast) and travellers who decide to finish early |

The willingness sweep compares travellers with low, mixed and high willingness to accept alternatives.

Metrics: feasibility, realised satisfaction, preferred activities completed, waiting time, crowd exposure, share of visits made while a place was over capacity, peak load, load inequality across attractions, CO₂, walk + transit share, local-business visits, accepted suggestions and replans. Every system sees the same travellers, background crowds and sensor noise for a given seed.

## Results at a glance

These come from the quick evaluation bundled in `outputs/` (run on macOS): 120 travellers per day, 2 seeds per scenario, mixed willingness. Each cell compares the conventional planner (satisfaction only) with the full sustainable system. `./run.sh all` reproduces them. Everything is seeded, so a rerun on the same machine gives the same numbers. macOS (libc++) and Linux (libstdc++) builds break some ties differently, so their numbers differ slightly, but no conclusion changes.

| Scenario | Satisfaction | Queue min / traveller | Overcrowded visits | CO₂ | Local businesses | Feasible |
| --- | --- | --- | --- | --- | --- | --- |
| S0 normal weekday | +1.9% | 4.7 → 0.1 | 13.7% → 3.6% | −69% | 1.16 → 1.76 | 96% → 100% |
| S1 peak weekend | +5.5% | 8.9 → 0.5 | 27.9% → 8.4% | −69% | 1.16 → 1.73 | 95% → 100% |
| S2 sudden crowd | +3.0% | 12.9 → 6.2 | 21.5% → 8.6% | −69% | 1.16 → 1.74 | 90% → 95% |
| S3 public event | +6.2% | 6.6 → 0.7 | 24.0% → 9.6% | −69% | 1.16 → 1.77 | 80% → 100% |
| S4 transport delays | −1.1% | 4.5 → 0.1 | 16.7% → 4.9% | −67% | 1.15 → 1.72 | 93% → 98% |
| S5 capacity limits | +18.1% | 34.9 → 9.6 | 34.6% → 12.4% | −67% | 1.16 → 1.81 | 65% → 95% |
| S6 rain, plan changes | +2.7% | 6.6 → 0.5 | 17.0% → 6.2% | −68% | 1.16 → 1.73 | 92% → 100% |

What the components contribute:

- **Ablations.** Without the Bayesian forecast, overcrowded visits double on the peak weekend (8.4% → 16.1%). Without replanning, feasibility on the event day drops from 100% to 88%. Replacing the learned policy with a simple crowd rule raises queueing from 0.5 to 0.7 min and overcrowded visits from 8.4% to 9.1%.
- **RL (in its own environment).** LinUCB, Q-learning and SARSA beat the satisfaction-only, minimum-distance, environment-only, rule-based and random policies.
- **Forecasting (held-out days).** The Bayesian network's load error is 0.036 vs 0.054 for historical averages. An unannounced surge is detected about 30 minutes after it starts.
- **Search.** A* matches UCS's optimal cost while expanding 43% fewer nodes; greedy best-first is 30% costlier on average.
- **Willingness.** As travellers become more willing, accepted suggestions rise from 25% to 58%. Even reluctant travellers see overcrowding halve (24% → 13%).

The trade-off: travellers complete slightly fewer of their personal top-5 sights (about 2.9 vs 3.4 per day), but overall satisfaction is higher because they spend far less time in queues and crowds.

## Project layout

```
run.sh                      one entry point for everything
config/default.json         all parameters: weights per system, RL, evaluation grid
cpp/include, cpp/src        C++17 engine: json, city, search (M1), csp (M1), planner (M2), engine, main
cpp/tests/test_engine.cpp   engine unit tests
tourism/core                shared types, config, city model
tourism/data                synthetic city, crowd history, traveller personas
tourism/engine              JSON-lines client for the C++ engine
tourism/m1_search           utilities, candidate selection, engine wrappers
tourism/m2_planning         planning domain description and PDDL export
tourism/m3_uncertainty      Bayesian network, forecaster, online updating, risk
tourism/m4_rl               features, environment, agents, training
tourism/coordinator         seams for GenAI, rule-based coordinator, validator, explainer
tourism/genai               tool registry and agent stubs (Module 5, later)
tourism/sim                 truth model, scenarios, simulated travellers, day simulator
tourism/eval                workspace, experiment runner, HTML report
tests/                      Python unit tests
examples/profiles/          example travellers (culture, family with wheelchair, foodie, impossible request)
docs/                       architecture and GenAI integration notes
data/, build/, outputs/     generated (safe to delete; ./run.sh clean)
```

## Notes

- **The city is synthetic, so the project runs offline.** Its JSON schema mirrors OpenStreetMap POIs and streets plus GTFS transit lines. To use a real city, write `data/city.json` in the same schema and a `history.csv.gz` with the same columns.
- **Each simulated traveller is a party of 15 visitors** (`simulation.traveller_weight`), so a few hundred planned itineraries create realistic crowd pressure.
- **Everything is seeded.** Re-running gives the same numbers.
- **Troubleshooting:** "engine not found" means you should run `./run.sh build`. To use a specific Python or compiler, run `PYTHON=/usr/local/bin/python3 CXX=g++ ./run.sh all`. To limit the evaluation to one core, run `./run.sh evaluate --workers 1`.
