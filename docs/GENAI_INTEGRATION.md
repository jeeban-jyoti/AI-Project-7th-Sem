# Adding Module 5 (Generative AI) later

The core system is complete without GenAI. The agents are added by replacing three stand-ins and wrapping modules 1–4 as tools. The middle of the pipeline (CSP, planner, forecaster, RL policy, validator) does not change.

## Where each agent plugs in

| Agent (from the brief) | Replaces or adds | Calls (already in `tourism/genai/tools.py`) |
| --- | --- | --- |
| Traveller | Replaces `FormProfileBuilder` (chat to `TravellerProfile`, clarifying questions, unsolvable requests). Answers `TravellerInterface.respond` and `reveal_willingness` (the RL `ask` action), which `SimulatedTraveller` / `ConsoleTraveller` answer today | Profile builder and the `ask` hook |
| Attraction | Adds plain-language monitoring of hours, capacity and crowds | `poi_info`, `crowd_forecast` |
| Transport | Adds a mode comparison for each leg | `compare_modes`, `route` |
| Sustainability | Adds the environmental and social case for an option | `impact`, `crowd_forecast` |
| Local business | Adds restaurants, shops, guides and community experiences outside busy areas | `search_local`, `poi_info` |
| Coordinator | Replaces `RuleCoordinator` (negotiation) and `TemplateExplainer` (explanations) | `plan_day`, `rl_suggest`, `validate` and every tool above |

The stubs are in `tourism/genai/agents.py` and implement the same interfaces as the stand-ins (`ProfileBuilder`, `TravellerInterface`, `Coordinator`). Swapping one in is a one-line change where `Workspace.coordinator()` or the CLI builds the stand-in.

## Rules already followed so the swap stays clean

1. Modules talk only through the shared types in `tourism/core/types.py` and JSON at the engine boundary, so they work as tools unchanged.
2. Every coordinator implements `Coordinator.plan(profile, traveller)` and `Coordinator.replan(...)`.
3. Profiles enter through one door: `ProfileBuilder.build_profile`.
4. The explainer reads only `ReasonRecord`s, each holding the change, the reason and the metric deltas, so an LLM can phrase facts it did not invent.
5. `validate_plan` is mandatory: no plan from any coordinator reaches the traveller without passing it.
6. The RL `ask` action goes through `TravellerInterface.reveal_willingness()`.

## Suggested order

1. Build the traveller agent first. It should produce the same `TravellerProfile` as the form on a set of test requests, including the `impossible_request.json` style of conflict.
2. Build the coordinator agent next. On the same profiles it must match or beat `RuleCoordinator`'s metrics and always pass the validator. Check that every number in its explanation appears in the `ReasonRecord`s.
3. Add the specialists, each limited to its own tools.
4. Re-run `./run.sh evaluate`. Explanation quality (clarity, faithfulness and persuasiveness) is the one metric that waits for this module. The `acceptance.explanation_boost` parameter is where measured nudging effects (e.g. the decoy effect from Chvirova et al.) can replace the assumed value.

`./run.sh tools` lists the tool schemas and calls a few of them.
