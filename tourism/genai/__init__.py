"""Module 5 (Generative AI) - hooks only for now.

Nothing here calls an LLM yet. What exists:
- tools.py   a working tool registry (name, JSON schema, function) over modules 1-4,
             so a future agent can call `route`, `crowd_forecast`, `solve_itinerary`,
             `plan`, `replan`, `validate`, `rl_suggest`, ... exactly as the
             rule-based coordinator does today;
- agents.py  the six agent classes from the brief as typed stubs, each saying
             which stand-in it replaces and which tools it will use.
See docs/GENAI_INTEGRATION.md for the integration plan.
"""
