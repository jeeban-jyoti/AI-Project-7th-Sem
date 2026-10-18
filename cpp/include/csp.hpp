// Module 1 (CSP): itinerary construction as a constraint satisfaction problem.
// Variables: one start time per candidate activity (or SKIP).
// Hard constraints: opening hours, no overlap incl. travel, day window, budget,
// max walking, accessibility, capacity, closures, must-visits, lunch, pace.
// Soft constraints (penalties): crowd threshold, waiting, CO2 target, transfers.
// Solver: greedy incumbent + depth-first branch & bound with MRV, degree and
// least-constraining-value ordering, forward checking and AC-3 preprocessing.
#pragma once

#include <map>
#include <string>
#include <unordered_map>
#include <vector>

#include "city.hpp"
#include "search.hpp"

namespace tp {

struct Profile {
    int start_node = 0, end_node = 0;
    int start_min = 540, end_min = 1200;
    double budget = 5000, max_walk_m = 8000;
    bool accessible_only = false;
    int modes = kAllModes;
    int max_activities = 5;
    bool lunch = true;
    int lunch_from = 720, lunch_to = 870;
    double crowd_threshold = 0.85, co2_target_kg = 1e9, hard_capacity = 1.6;
    int max_transfers = 99;

    static Profile fromJson(const Json& j);
};

int modesFromJson(const Json& arr, int def);

struct Candidate {
    int node = 0, poi = 0;
    double utility = 0, price = 0;
    int dur = 60;
    bool must = false, restaurant = false, local = false;
    int modes_override = -1;  // modes allowed for the leg arriving here (-1 = profile modes)
    std::vector<std::pair<int, int>> forbid;  // forbidden start windows [from, to)
};

// Cached leg routing shared by CSP and planner. Travel costs only change with
// the time "regime" (peak / off-peak and which delays are active), so legs are
// cached per (from, to, regime, modes) and filled with one-to-many searches.
class LegRouter {
public:
    struct Leg {
        bool found = false;
        double time = 0, cost = 0, money = 0, co2 = 0, walk = 0, dist = 0;
        int transfers = 0;
        bool car = false;
    };
    LegRouter(const City& city, const Profile& prof, const Weights& w, const Conditions* cond)
        : city_(city), router_(city), prof_(prof), w_(w), cond_(cond) {}

    void addTargets(const std::vector<int>& nodes);
    const Leg& leg(int from, int to, double depart, int modes);
    RouteResult full(int from, int to, double depart, int modes) const;
    std::size_t cacheSize() const { return cache_.size(); }
    long searches() const { return searches_; }

private:
    RouteRequest makeReq(int from, int to, double depart, int modes) const;
    int regimeOf(double t) const;
    static long long key(int from, int to, int regime, int modes) {
        return ((static_cast<long long>(from) * 8192 + to) * 256 + regime) * 8 + modes;
    }
    const City& city_;
    Router router_;
    const Profile& prof_;
    Weights w_;
    const Conditions* cond_;
    std::vector<int> targets_;
    std::vector<char> isTarget_;
    std::unordered_map<long long, Leg> cache_;
    long searches_ = 0;
};

struct CspOptions {
    long node_limit = 60000;
    double time_limit_ms = 400;
    int value_limit = 8;
    bool soft_as_hard_crowd = false;
    bool soft_as_hard_co2 = false;
    std::vector<std::string> relax_order{"co2_target", "crowd_threshold"};
};

Json solveItinerary(const City& city, const Context* ctx, const Json& request);

}  // namespace tp
