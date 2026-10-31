# Sustainable and Crowd-Aware Tourism Planner

AI course project (Group 15). The planner builds a one-day sightseeing plan for a tourist. Like any
trip planner, it tries to match the traveller's interests. It also tries to:

- avoid crowds and long queues,
- prefer walking and buses over taxis (less CO₂),
- send some visitors to quieter places and local businesses, so the famous sights are not overloaded.

The project covers five AI topics, one Python file each:

| Module | AI topic | File |
| --- | --- | --- |
| 1 | Search (UCS, Greedy, A*) and Constraint Satisfaction (CSP) | `module1_search_csp.py` |
| 2 | Classical planning (actions with preconditions and effects), replanning, plan validation | `module2_planning.py` |
| 3 | Uncertainty: Bayesian network, Bayes' rule, Monte Carlo simulation | `module3_uncertainty.py` |
| 4 | Reinforcement learning: Q-learning and SARSA | `module4_rl.py` |
| 5 | Generative AI: plain-language requests, the guided trip form, explaining plans | `module5_genai.py` |
| - | Connects all modules: demo, interactive trip planner, comparison | `main.py` |
| - | Builds the city and every dataset | `generate_data.py` |
| - | Tests for the datasets and every module | `test.py` |

Everything uses plain Python (3.8 or newer) with no packages to install.

---

## 1. How to run it

```bash
python3 main.py interactive  # plan YOUR day: choose from options, get a plan, try "what if"
python3 main.py              # full demo: every module, step by step (about 2 seconds)
python3 test.py              # interactive test menu
```

The first run creates the datasets in `data/` (under a second). To make them yourself, or a
different city with another seed:

```bash
python3 generate_data.py
python3 generate_data.py --seed 8
```

### The interactive trip planner

`python3 main.py interactive` (or `python3 main.py plan --interactive`, as in the first version)
asks 12 questions. You answer each one with the number of an option, and Enter takes the suggestion:

| # | Question | Options |
| --- | --- | --- |
| 1 | Where are you staying? | the 6 hotels |
| 2-4 | Which day, which season, what does the forecast say? | weekday / weekend; off / peak; sunny / cloudy / rainy |
| 5 | What do you enjoy? (pick several, favourite first) | history, art, nature, temples, architecture, family, food, shopping, crafts |
| 6 | Any place you must see? | the 10 best places for your interests, each with rating, price, opening hours and the **chance it is crowded at midday** |
| 7-8 | When do you leave, when must you be back? | a list of times, or type one |
| 9 | Budget | 1000 / 2000 / 3000 / 5000, or type one |
| 10-11 | Transport, wheelchair access | walking + buses, or with taxis; yes / no |
| 12 | Pace | relaxed (3 places), normal (4), packed (5), or no lunch break |

You then get your plan, next to the plan a crowd-blind planner would make, with an explanation, the
risk of things going wrong, and a check by the independent validator. If the learned policy suggests a
change, you accept or reject it. After that, a menu lets you:

1. keep the plan and finish,
2. change your answers and plan again,
3. pretend one of your places is closed when you arrive (the rest of the day is replanned),
4. pretend your bus or taxi is delayed by 20, 40 or 60 minutes (replanned),
5. see a quieter alternative for each stop, with both crowd forecasts.

### Other commands

| Command | What it shows |
| --- | --- |
| `python3 main.py plan culture` | A full plan for a sample traveller (`culture`, `family`, `foodie`, `impossible`) |
| `python3 main.py ask "Lakeside Resort, wheelchair, we love nature, budget 3000, from 10 to 6"` | Plan from a plain-language request |
| `python3 main.py search` | UCS vs Greedy vs A* on five routes |
| `python3 main.py forecast` | Bayesian network accuracy, Brier score and crowd forecasts |
| `python3 main.py rl` | Trains Q-learning and SARSA and compares them with fixed policies |
| `python3 main.py compare` | Conventional vs sustainable planner for the first 30 travellers in the dataset (about 10 seconds) |

Options that work with every command:

- `--interactive` lets you accept or reject the suggestions yourself.
- `--day weekday|weekend`, `--season off|peak` and `--rain 0.6` set the day.
- `--llm` uses Claude for Module 5. This needs `pip install anthropic` and an `ANTHROPIC_API_KEY`.
  Without them, the project uses its built-in rules.

### Tests

`test.py` is plain Python. Every test prints what it checks, the values it found, and `PASS` or
`FAIL`, for example:

```
  [4/17] A* finds the same cost as UCS and expands no more nodes
      co2_weight = 1 (greener route)
      PASS  Central Hotel -> Hill Monastery        UCS   66.4/148 nodes   A*   66.4/120 nodes
```

```bash
python3 test.py          # menu: pick a group, step through tests, or try your own input
python3 test.py all      # all 75 tests (331 checks) without questions, about 15 seconds
python3 test.py 3        # one group: 0 = datasets, 1-5 = modules, 6 = all modules together
python3 main.py test     # same as: python3 test.py all
```

| Key in the menu | What it does |
| --- | --- |
| `0`-`6` | Run one group of tests |
| `a` | Run all tests, then show a summary per group |
| `s` | Step through all tests: Enter after each one, `q` to stop |
| `t` | **Try your own input**: you type the values and the program checks the result |

"Try your own input" has six interactive tests:

| Test | You choose | It checks |
| --- | --- | --- |
| Plan my whole trip | Everything, in the interactive trip planner above | Your final plan passes the validator: opening hours, tickets, back at the hotel in time |
| Route | Start, goal, fastest or greener | A\* cost = UCS cost, A\* expands fewer nodes, Greedy is not cheaper, the heuristic is admissible |
| Trip in plain words | Any sentence | Every hard constraint and planner precondition, on your plan |
| Crowd forecast | Place, time, day, season, rain, then a live sensor reading | Probabilities sum to 1; the sensor reading raises the probability of what it reported |
| RL policy | Crowd level, time, whether a quieter place exists | The chosen action is allowed and has the highest Q value |
| Replanning | A traveller, then "a place closes" or "a delay", and when | The closed place is skipped; the traveller is back at the hotel in time |

| Group | Tests | What they check |
| --- | --- | --- |
| 0 Datasets | 10 | The generator is repeatable; `data/` matches it; the city is as large as the first version's; every place has valid values; every place is reachable (wheelchair users too); the history has the right rows and realistic patterns; queues build and drain; buses are later in rain and rush hour; all 300 travellers are valid |
| 1 Search and CSP | 17 | A\* = UCS cost with fewer nodes (5 routes and 25 random pairs); Greedy is never cheaper; the heuristic is admissible; wheelchair routes avoid steep paths; the greenest route avoids taxis; a suspended bus line is not used; every itinerary obeys every hard constraint; impossible must-visits are explained |
| 2 Planning | 10 | The plan reaches the goal; preconditions hold; the three replanning stages work; the validator catches 11 kinds of broken plan and passes every real plan |
| 3 Uncertainty | 15 | Inference matches a hand calculation (sprinkler network); soft evidence; learning recovers known probabilities from 20,000 samples; rain moves people indoors; a peak weekend is busier; the network beats the baseline in accuracy and Brier score; the sensor detects a sudden crowd; less slack means more risk |
| 4 RL | 9 | The 24 states and their allowed actions; Q-learning and SARSA updates match a hand calculation; Q-learning finds the best action from noisy rewards; crowding lowers the reward; the learned policies beat doing nothing and random; results repeat exactly |
| 5 GenAI | 8 | Parsing of requests; the explanation; a bad Claude answer is cleaned up (fake answer, no internet); fallback without Claude; the guided form gives the right profile, and wrong answers fall back to the suggestion |
| 6 Together | 6 | Every sample traveller gets a plan; accepted and rejected suggestions; load inequality (Gini); a full day for 15 travellers from the dataset; the interactive trip planner end to end |

**The first version's tests are included.** Every test from `tests/*.py` and
`cpp/tests/test_engine.cpp` on the `main` branch has a matching test here, marked "(original test)":

| First version | Here |
| --- | --- |
| `test_bayes_net.py`: inference vs enumeration, soft evidence, learning recovers the tables | `bn_enumeration_by_hand`, `bn_soft_evidence_prior`, `bn_learning_recovers` |
| `test_forecaster.py`: normalised, context matters, beats the historical average | `forecast_context`, `network_beats_baseline` |
| `test_online.py`: detects a surge, not a normal day | `sensor_detects_surge` |
| `test_engine_integration.py`: A\* on 25 random pairs, valid plans for generated travellers, impossible request | `astar_random_pairs`, `validator_on_dataset`, `csp_opening_hours_conflict` |
| `test_rl.py`: state space, Q-learning finds the best action | `rl_state_space`, `rl_bandit` (LinUCB is not in this version, so its test is not either) |
| `test_simulation.py`: queues, Gini, crowding lowers satisfaction, a full day runs | `data_queue`, `gini_values`, `rl_crowding_hurts`, `full_day_dataset` |
| `test_validator.py`: the validator catches violations | `validator_catches_violations` |
| `test_engine.cpp`: JSON round trip, wheelchair path, greenest avoids the car, suspended line, CSP, replanning | `data_files_up_to_date`, `wheelchair_route`, `greenest_avoids_taxi`, `suspended_line`, `csp_constraints`, `replan_closed` |

**Adding a test:** write a function with a one-line docstring, put `@test(group_number)` above it, and
use `show(label, value)` to print values and `check(description, condition)` for each check. It then
appears in the menu automatically.

---

## 2. What happens when you plan a day

```
 Guided form, or "We love nature, wheelchair, ..."   <- Module 5 turns your choices into a profile
                     |
                     v
 Day context (weekend, peak season, 30% rain)        <- Module 3 forecasts crowds for every
                     |                                  place and hour (Bayesian network)
                     v
 CSP picks places, order and times                   <- Module 1 (routes between places by A*)
                     |
                     v
 For each stop: keep / shift / substitute?           <- Module 4 (learned Q-table); the traveller
                     |                                  accepts or rejects; the CSP is re-solved
                     v
 Timed action plan: travel, ticket, wait,            <- Module 2 (A* over world states)
 visit, eat, rest
                     |
                     v
 Independent check of every rule                     <- Module 2 (validate_plan)
 Chance the plan fails (queues, bus delays)          <- Module 3 (Monte Carlo), plan B if > 30%
                     |
                     v
 Friendly explanation                                <- Module 5
                     |
 During the day: a place closes / a delay -> Module 2 replans the rest of the day
```

Every plan is also made by a **conventional** planner that only cares about interests and fame. This
baseline uses taxis and ignores crowds. Comparing the two shows what the sustainable planner gains.

---

## 3. The data (`generate_data.py`)

| File | Size | What it is |
| --- | --- | --- |
| `data/city.json` | 48 KB | **The city**: 73 places in 8 districts: 35 sights, 10 local businesses, 14 restaurants, 6 hotels and 8 bus stations. Each place has a position (km), categories, popularity, rating, how many visitors it can take per hour, visit time, price, opening hours, indoor/outdoor, wheelchair access and whether it needs a ticket. Also 352 footpaths (12 steep ones in hilly Hill Park, not for wheelchairs), 4 bus lines, and the speed, CO₂ and fare of walking, bus and taxi. |
| `data/history.csv` | 5.5 MB | **Visitor history**: 120 days × every opening hour × 59 places = 65,040 rows. Columns: `day, Day, Season, Weather, hour, Block, place, Fame, Setting, event_zone, visitors, Crowd, wait_min, Wait`. Module 3 learns its Bayesian network from this. |
| `data/transit_history.csv` | 0.3 MB | **Bus delays**: 120 days × 4 lines × 13 hours = 6,240 rows: `day, line, hour, Weather, peak, event_on_line, delay_min, BusDelay`. Module 3 learns P(bus late \| weather) from this. |
| `data/travellers.json` | 0.1 MB | **300 travellers** of 7 kinds (culture, foodie, nature, family, shopper, spiritual, explorer), with hotel, times, budget, interests, pace, transport, wheelchair needs, sometimes a must-visit, and a hidden "flexibility" (how often they accept a suggestion). Used by `main.py compare` and the tests. |

This is the same size as the first version's city: 59 places to visit, 4 lines, 120 days of history.
The history is hourly rather than every 15 minutes, which keeps the file at a few MB instead of
hundreds of thousands of rows.

**How the history is made.** The data is **synthetic**, so the project runs anywhere without
downloads. A hidden "true" model (the `true_*` functions in `generate_data.py`) decides how full each
place is in every hour:

```
load = DEMAND × base(popularity) × day factor × season factor × rain factor × hour curve × event × noise
```

- **Day:** weekends are busier.
- **Season:** peak season is busier.
- **Rain:** outdoor places get emptier and indoor places fuller.
- **Hour curve:** sights peak late morning and mid-afternoon, restaurants at lunch and dinner.
- **Events:** a public event on 15% of days makes one district 60% busier for a few hours.

Then:

- **Visitors** = load × capacity.
- **Crowd level** = low below 55% full, medium below 85%, high above that.
- **Queue**: people who cannot get in during one hour wait into the next (a fluid queue), plus a
  little extra crowding at the entrance.
- **Bus delays**: an 8% chance of a short delay and a 2% chance of a long one. Rain doubles both
  chances; rush hour and events make them 1.5 times as likely.

The Bayesian network never sees this model, only the CSV files, just like a real system that only has
visitor logs. That is what makes it fair to measure how well it learned.

**Everything is seeded**: the same seed always gives byte-identical files (a test checks this). To
change the city, edit the tables at the top of `generate_data.py` (places, districts, bus lines) and
run `python3 generate_data.py`.

---

## 4. The modules in detail

### Module 1: Search and CSP (`module1_search_csp.py`)

**Part A: Search.** The city is a graph. Places are nodes. Edges are:
- footpaths (from `city.json`),
- taxi rides (places closer than 3 km),
- bus links (consecutive stations of a bus line).

Each edge stores minutes, CO₂ grams, fare and distance.

- Cost of an edge = `minutes + co2_weight × CO₂ / 10`. With `co2_weight = 0` you get the fastest route;
  with `1` you get a greener one.
- `search(city, graph, start, goal, method)` runs:
  - **UCS**: priority = g (cost so far)
  - **Greedy best-first**: priority = h (estimate to goal)
  - **A\***: priority = g + h
- The heuristic h = straight-line distance at the fastest speed (taxi). It never over-estimates, so
  A\* is optimal.
- `build_graph(..., accessible_only=True)` leaves out steep footpaths and places without step-free
  access, for wheelchair users. `suspended_lines=[...]` leaves out a bus line, for example during a
  strike.

**Part B: CSP.** `solve_itinerary(city, profile, system, forecast)` chooses the day's stops.

- **Variables**: stop 1, stop 2, … in time order.
- **Domains**: the 8 best sights and 4 best restaurants for this traveller. Places near the hotel are
  preferred, because the city is about 8 km wide.
- **Hard constraints**: open during the whole visit, back at the hotel in time, within budget, lunch
  starts 11:30-14:30, each place once, wheelchair access, all must-visit places included.
- **Soft constraints** (score): interest match, crowd forecast, queue, travel time and CO₂, bonus for
  local businesses and quiet places.
- **Solver**: backtracking search with
  - *node consistency* (remove places that break a single-place constraint before starting),
  - *forward checking* (stop a branch as soon as a must-visit or lunch can no longer fit),
  - *value ordering* (try the best-scoring places first),
  - *branch and bound* (skip a branch whose best possible score cannot beat the best plan found).
- If the must-visit places cannot all fit, it explains why ("Hill Monastery is not wheelchair
  accessible", "Night Food Street is open 17:00-21:00, outside your day") and relaxes them one at a
  time.
- The two planning systems are just two weight settings in `SYSTEMS`. The sustainable one also keeps
  30 minutes spare for queues and delays.

### Module 2: Planning, replanning and validation (`module2_planning.py`)

The CSP decides **which** places. The planner decides **how**: a sequence of timed actions.

- **State**: (location, time, energy, money, places visited, lunch eaten, tickets held)
- **Actions**, each with preconditions and effects (STRIPS style):

  | Action | Preconditions | Effects |
  | --- | --- | --- |
  | `travel(a, b)` | at a, money for fare, energy to walk, no unused ticket for a | at b, time, money, energy and CO₂ change |
  | `buy_ticket(p)` | at p, p needs a ticket, enough money | ticket for p, money − price |
  | `wait(p)` | at p, p not open yet | time = opening time |
  | `visit(p)` | at p, open for the whole visit, ticket if needed, money for any door fee, energy ≥ 20 | visited p, energy − 12 per hour, money − door fee |
  | `eat(r)` | at restaurant r, not eaten, 11:30-14:30 | eaten, energy + 25 |
  | `rest` | energy < 40 | energy + 30, 20 minutes |

- **Goal**: all places visited, lunch eaten, back at the hotel before the end of the day.
- **Search**: `plan(problem)` runs A\* over states. Cost = time (+ CO₂ and crowd cost for the
  sustainable system). Heuristic = time still needed for the remaining visits and lunch, which is
  admissible.
- **Replanning**: `replan(problem, steps, now, event)` handles "a place closes" or "a 40-minute
  delay". It works in three stages:
  1. Keep the plan if the event does not affect it.
  2. Otherwise, replan the rest of the day with A\*.
  3. If even that fails, drop the least valuable place and try again.
- **Validation**: `validate_plan(city, profile, steps)` re-checks a finished plan from scratch,
  without trusting the planner, as in the first version. It looks for:
  - wrong transport, wrong place, overlapping steps,
  - opening hours, missing tickets, closed places, wheelchair access,
  - lunch time, places visited twice,
  - the end place and time, the budget, and missing must-visits.

  Every plan shown to a traveller goes through it.

### Module 3: Uncertainty (`module3_uncertainty.py`)

**Bayesian network:**

```
Day, Season, Weather, Time block, Fame of place, Indoor/outdoor  --->  Crowd  --->  Queue
Weather  --->  Bus delay
```

- **Learning**: `learn_network(history, transit)` counts how often each value happens for every
  combination of parents in the two CSV files, with Laplace (+1) smoothing.
- **Inference**: `BayesNet.query(var, evidence)` computes P(var | evidence) by enumeration, summing
  over the unknown variables. A forecast like "30% rain" is used as **soft evidence**.
- **Forecast**: `make_forecast()` gives Modules 1 and 2 a function `forecast(place, time)` returning
  (P(crowded), expected queue minutes).
- **Sensor update**: `sensor_update()` applies Bayes' rule with a live crowd sensor that is right 80%
  of the time.
- **Monte Carlo risk**: `plan_risk()` simulates the plan 500 times with random crowds, queues and bus
  delays. It reports the chance that the plan fails (a place closes before the visit ends, or the
  traveller gets back late). If that chance is above 30%, `main.py` offers a shorter plan B.
- **Evaluation**: `evaluate_forecast()` trains on the first 90 days and tests on the last 30. The
  baseline knows only each place's usual crowd levels. Two scores are reported: accuracy, and the
  Brier score, which measures how good the probabilities are.

### Module 4: Reinforcement learning (`module4_rl.py`)

Many travellers want the same famous places at the same time. For each stop in a traveller's plan, the
agent decides whether to suggest a change.

- **State** (24 states): crowd level (low/medium/high) × time block (4) × is there a similar quieter
  place nearby (yes/no)
- **Actions**:
  - `keep`
  - `shift` (go in the next time block)
  - `substitute` (a similar, less popular place within 2.5 km)
- **Reward**: enjoyment − 8 × (how far the place is over 80% full) + 0.5 for a less-visited place
  − 0.3 if the traveller says no
- **Environment** (`CrowdEnv`): one episode = one day with 40 travellers and 3 stops each. Each
  traveller has a hidden willingness to accept suggestions. Every choice adds load to that place, so
  later travellers face the crowds that earlier decisions created.
- **Algorithms**: `TabularAgent` with Q-learning (target uses the best next action) or SARSA (target
  uses the action actually taken next). It uses ε-greedy exploration, falling from 1.0 to 0.05, and a
  learning rate that shrinks as a state-action pair is visited more.
- **Compared with**: always keep, random, and a hand-written rule (if very crowded, substitute or
  shift).

### Module 5: Generative AI (`module5_genai.py`)

- `ask_profile(city)` is the **guided form** of the interactive trip planner: 12 numbered questions
  that build the traveller profile. Wrong answers fall back to the suggested option.
- `parse_request(text, city)` turns a sentence into a traveller profile. It picks out:
  - interests (keywords),
  - hotel and must-visit places (names from the city),
  - wheelchair access, "no taxi", budget and hours (regular expressions).
- `explain_plan(facts)` turns the numbers from all modules into a short explanation.
- With `--llm`, parsing and explaining ask Claude (`claude-opus-5-5`, through the official `anthropic`
  SDK). Claude's JSON answer is **checked field by field** before use. Anything invalid is ignored,
  and the rules take over if Claude is unavailable. The request turns on the API's automatic fallback
  model in case a request is declined.

---

## 5. Results

These are the outputs of the commands above, on a peak-season weekend with a 30% chance of rain.
Everything is seeded, so you get the same numbers.

**Search (`main.py search`, cost = minutes + CO₂/10):** A\* finds exactly the same optimal cost as UCS
on every route. On five routes it expands 103 nodes on average against 120 for UCS; on 25 random pairs
it expands 19% fewer. Greedy expands only 1-3 nodes, but its routes cost 12-66% more.

**Bayesian network (`main.py forecast`)**, on 30 held-out days (16,260 place-hours):

| | Network | Per-place average |
| --- | --- | --- |
| Accuracy (right crowd level) | **86.5%** | 80.1% |
| Brier score (lower is better) | **0.191** | 0.275 |

A live "high" sensor reading at Royal Fort at 10:00 raises P(crowded) from 0.34 to 0.81.

**Reinforcement learning (`main.py rl`, 60 test days):**

| Policy | Reward per decision | Overcrowded visits | Enjoyment | Suggestions accepted |
| --- | --- | --- | --- | --- |
| Always keep | 4.594 | 9.0% | 5.00 | - |
| Random | 4.499 | 5.7% | 4.87 | 38% |
| Simple rule | 4.655 | 4.9% | 4.96 | 34% |
| Q-learning | 4.658 | 4.8% | 4.96 | 35% |
| SARSA | 4.659 | 4.8% | 4.96 | 35% |

By trial and error, the learners find a policy as good as the hand-written rule (very slightly better)
and almost halve overcrowding compared with doing nothing. What they learned: substitute at midday when
a quieter similar place exists, shift busy afternoon visits to the evening, and leave mornings and
evenings alone.

**Whole system (`main.py compare`, the first 30 travellers in `data/travellers.json`):**

| Average per traveller | Conventional | Sustainable |
| --- | --- | --- |
| Interest score (higher is better) | 43.2 | 41.1 |
| Queue minutes | 54.0 | 16.7 |
| Visits at a crowded place | 1.83 | 0.21 |
| CO₂ grams | 1373 | 303 |
| Travel minutes | 58 | 131 |
| Chance the plan fails | 32% | 2% |
| Money spent | 1116 | 907 |
| Load inequality across places (Gini, lower = more even) | 0.74 | 0.70 |
| Different places visited (all 30 travellers) | 21 | 25 |

All 30 travellers got a plan from both systems. The independent validator found no problems in any of
our plans.

**Honest observations** (good to mention in the report and viva):

- **The cost of being sustainable.** Travellers get a slightly lower interest score (−5%) and spend
  about twice as long travelling, because they walk and take buses instead of taxis. In return they:
  - queue 69% less,
  - meet crowds 89% less often,
  - emit 78% less CO₂,
  - see the chance of a broken plan fall from 32% to 2%.
- **Visits are spread only a little more evenly** (Gini 0.74 → 0.70). Each traveller is planned on
  their own. Planning many travellers together (the collective effect inside Module 4's environment)
  would spread them more; see the extensions.
- **The RL agent rarely acts on our own plans.** The crowd-aware CSP already avoids crowded slots, so
  on those plans the agent keeps every stop: zero suggestions for the 30 travellers. It does act on
  crowd-blind plans. In the demo it suggests Old Town Pottery Workshop instead of National Art Gallery
  at 13:06, and moving the Royal Fort visit to after 17:00.
- **Results come from a synthetic city.** They show that the methods work together, not how a real
  city would behave.

---

## 6. Team split (5 students)

| Student | Owns | Should be able to explain |
| --- | --- | --- |
| 1 | `module1_search_csp.py`, the city part of `generate_data.py` | Graph building, UCS vs Greedy vs A\*, admissible heuristic, wheelchair routes, CSP variables/domains/constraints, forward checking, branch and bound |
| 2 | `module2_planning.py` | States, actions with preconditions/effects, A\* planner, heuristic, the three replanning stages, the validator |
| 3 | `module3_uncertainty.py`, the history part of `generate_data.py` | The hidden model and the data it produces, Bayesian network structure, learning CPTs with Laplace smoothing, enumeration, soft evidence, Bayes' rule for sensors, Brier score, Monte Carlo |
| 4 | `module4_rl.py` | MDP (states, actions, reward), Q-learning vs SARSA update, ε-greedy, the comparison table |
| 5 | `module5_genai.py` + `main.py` | The guided form, rule-based parsing and templates, how Claude is used safely (validation, fallback), how `main.py` connects the modules, the interactive trip planner |

Each file starts with a docstring describing it. Read those first. Each student also owns the
matching tests in `test.py` (`python3 test.py 0` for the datasets, `1` … `5` for the modules). After
changing a file, run its tests.

**Easy things to try** (good for showing that you understand your part):

- Module 1: change `co2_weight` or the weights in `SYSTEMS` and see how routes and plans change.
- Module 2: change `VISIT_ENERGY_PER_HOUR` so that the planner has to insert `rest` actions.
- Module 3: change `DEMAND` or the day, season and rain factors in `generate_data.py`, run
  `python3 generate_data.py`, and see the network learn the new pattern.
- Module 4: change the reward weights in `CrowdEnv.step`, or `gamma`, and compare the tables.
- Module 5: add keywords to `INTEREST_WORDS`, or a new question to `ask_profile`.

---

## 7. What changed from the first version

The first version of this project, still on the `main` branch, did the same job at a much larger
scale:
- a C++17 engine for search, CSP and planning, driven from Python over JSON,
- a 59-place generated city with 4 transit lines,
- a Bayesian network with hierarchical smoothing and online Gamma-Poisson updating,
- three RL algorithms (Q-learning, SARSA and LinUCB),
- a city simulator with 7 scenarios (events, closures, transport delays, rain),
- multi-core evaluation with ablations and an HTML report.

That was more than five students could own and explain, so this version keeps the AI ideas, the data
size and the tests, and removes the engineering scale:

| First version | This version |
| --- | --- |
| C++ engine + Python package, about 90 files | 7 Python files + generated data |
| Needed a C++ compiler and `./run.sh` | Only `python3`, nothing to install |
| 59 places to visit, 4 lines, 120 days of 15-minute history | 59 places to visit, 4 lines, 120 days of hourly history, 300 travellers |
| CSP with MRV, degree, LCV, AC-3 in C++ | CSP with node consistency, forward checking, value ordering, branch and bound |
| Planner with PDDL export, min-conflicts repair, Python validator | A\* planner with readable actions; keep / replan / drop; Python validator |
| BN with hierarchical Dirichlet smoothing, online Gamma-Poisson | BN with Laplace smoothing, enumeration, Bayes' rule sensor update |
| Q-learning, SARSA, LinUCB, 1,536 states, 6 actions | Q-learning, SARSA, 24 states, 3 actions |
| `plan --interactive` form | Interactive trip planner with numbered options and what-if events |
| 7 simulated scenarios, ablations, HTML report, 10-20 min run | One comparison command, about 10 seconds |
| GenAI as stubs only | Working rule-based Module 5, with optional Claude |
| 7 Python test files + C++ tests | `test.py`: all of them ported, plus dataset and interactive tests (75 tests) |

To look at the first version:

```bash
git checkout main
```

---

## 8. Ideas for extensions

- Plan many travellers in a row and add each plan to the crowd forecast, to spread visits more evenly
  (the collective effect from Module 4's environment, inside the real planner).
- Use the `event_zone` column of the history as an extra parent of `Crowd` in the Bayesian network.
- Add a `switch_mode` action to Module 4, for example suggesting a bus instead of a taxi.
- Replace the tables in `generate_data.py` with real places from OpenStreetMap.
