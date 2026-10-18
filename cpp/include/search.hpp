// Module 1 (search): Uniform-Cost, Greedy Best-First and A* over a
// time-dependent multimodal graph (walk, transit lines, car/taxi).
#pragma once

#include <string>
#include <vector>

#include "city.hpp"

namespace tp {

enum class Algorithm { UCS, GREEDY, ASTAR };
Algorithm algorithmFromString(const std::string& s);

struct RouteRequest {
    int from = 0, to = 0;
    int depart = 540;
    int modes = kAllModes;
    bool accessible_only = false;
    Algorithm algorithm = Algorithm::ASTAR;
    Weights w;
    const Conditions* cond = nullptr;
};

struct Leg {
    int mode = WALK;
    int line = -1;
    int from = 0, to = 0;
    double depart = 0, arrive = 0;
    double dist_m = 0, money = 0, co2_kg = 0;
};

struct RouteResult {
    bool found = false;
    double cost = 0;
    double depart = 0, arrive = 0;
    double time_min = 0, money = 0, co2_kg = 0, walk_m = 0, dist_m = 0, crowd_exposure = 0;
    int transfers = 0;
    std::vector<int> path;
    std::vector<Leg> legs;
    long expanded = 0, generated = 0;
    double runtime_us = 0;
};

class Router {
public:
    explicit Router(const City& city) : city_(city) {}

    RouteResult route(const RouteRequest& req) const;
    // One-to-many: uniform-cost expansion until every target is settled.
    std::vector<RouteResult> routeToMany(const RouteRequest& base, const std::vector<int>& targets) const;

    double heuristic(int node, int goal, const RouteRequest& req) const;

private:
    struct Label {
        double g = 0, t = 0, money = 0, co2 = 0, walk = 0, dist = 0, crowd = 0;
        int boardings = 0;
        int parent = -1;
        int edge_mode = -1, edge_line = -1;
        double edge_dist = 0, edge_depart = 0;
    };
    struct SearchOutput {
        std::vector<Label> labels;
        std::vector<char> settled;
        long expanded = 0, generated = 0;
    };

    SearchOutput run(const RouteRequest& req, const std::vector<char>* targetMask, int nTargets) const;
    RouteResult extract(const SearchOutput& out, int goalNode, const RouteRequest& req) const;
    int nStateModes() const { return 2 + static_cast<int>(city_.lines.size()); }

    const City& city_;
};

Json routeToJson(const RouteResult& r, const City& city, bool withPath);
std::string modeName(const City& city, int mode, int line);

}  // namespace tp
