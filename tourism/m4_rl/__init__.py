"""Module 4: reinforcement learning for sustainable itinerary recommendation.

MDP: one decision per itinerary stop. Actions: keep, shift to off-peak,
substitute a similar less-crowded POI, switch the leg to transit / walking,
add a local business, or ask the traveller how open they are to compromises.
Algorithms: tabular Q-learning, SARSA and a LinUCB contextual bandit, compared
with keep-only (satisfaction), minimum-distance and environment-only policies.
"""
