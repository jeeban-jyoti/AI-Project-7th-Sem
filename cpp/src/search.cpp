#include "search.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <queue>
#include <stdexcept>
#include <functional>
#include <string>
#include <utility>
#include <vector>

namespace tp {

namespace {
constexpr double kInf = std::numeric_limits<double>::infinity();
constexpr int kFoot = 0;
constexpr int kCar = 1;
}  // namespace

Algorithm algorithmFromString(const std::string& s) {
    if (s == "ucs" || s == "uniform" || s == "dijkstra") return Algorithm::UCS;
    if (s == "greedy" || s == "gbfs") return Algorithm::GREEDY;
    if (s == "astar" || s == "a*") return Algorithm::ASTAR;
    throw std::runtime_error("unknown algorithm: " + s);
}

std::string modeName(const City& city, int mode, int line) {
    if (mode == WALK) return "walk";
    if (mode == CAR) return "car";
    if (line >= 0 && line < static_cast<int>(city.lines.size())) return city.lines[line].kind;
    return "transit";
}

double Router::heuristic(int node, int goal, const RouteRequest& req) const {
    if (goal < 0) return 0.0;
    double vmax = 1e-9;
    if (req.modes & (1 << WALK)) vmax = std::max(vmax, req.accessible_only ? city_.params.wheel_mpm : city_.params.walk_mpm);
    if (req.modes & (1 << TRANSIT)) vmax = std::max(vmax, city_.max_transit_mpm);
    if (req.modes & (1 << CAR)) vmax = std::max(vmax, city_.params.car_kmh_off * 1000.0 / 60.0);
    double d = city_.euclid(node, goal);
    // Every remaining edge costs at least w.time * dist / vmax (+ w.dist * km), all other terms are >= 0.
    return d * (req.w.time / vmax + req.w.dist / 1000.0);
}

Router::SearchOutput Router::run(const RouteRequest& req, const std::vector<char>* targetMask, int nTargets) const {
    const int S = nStateModes();
    const int N = static_cast<int>(city_.nodes.size());
    SearchOutput out;
    out.labels.assign(static_cast<std::size_t>(N) * S, Label());
    out.settled.assign(static_cast<std::size_t>(N) * S, 0);
    std::vector<double> best(static_cast<std::size_t>(N) * S, kInf);

    const bool singleGoal = targetMask == nullptr;
    const int goal = singleGoal ? req.to : -1;
    Algorithm alg = singleGoal ? req.algorithm : Algorithm::UCS;

    using QItem = std::pair<double, int>;
    std::priority_queue<QItem, std::vector<QItem>, std::greater<QItem>> pq;

    int start = req.from * S + kFoot;
    out.labels[start].t = req.depart;
    best[start] = 0;
    pq.push({heuristic(req.from, goal, req), start});
    out.generated = 1;

    const Conditions* cond = req.cond;
    const double congestion = cond ? cond->car_congestion : 1.0;
    std::vector<char> nodeSettled(N, 0);
    int settledTargets = 0;

    while (!pq.empty()) {
        int s = pq.top().second;
        pq.pop();
        if (out.settled[s]) continue;
        out.settled[s] = 1;
        ++out.expanded;
        const Label cur = out.labels[s];
        const int u = s / S, sm = s % S;

        if (singleGoal) {
            if (u == goal) break;
        } else if (!nodeSettled[u]) {
            nodeSettled[u] = 1;
            if ((*targetMask)[u] && ++settledTargets >= nTargets) break;
        }

        for (const Edge& e : city_.adj[u]) {
            if (!(req.modes & (1 << e.mode))) continue;
            double dt = 0, dmoney = 0, dco2 = 0, dwalk = 0, crowd = 0;
            int nsm = kFoot;
            bool boarding = false;
            if (e.mode == WALK) {
                if (req.accessible_only && (!e.accessible || !city_.nodes[e.to].accessible)) continue;
                dt = e.dist_m / (req.accessible_only ? city_.params.wheel_mpm : city_.params.walk_mpm);
                dwalk = e.dist_m;
            } else if (e.mode == TRANSIT) {
                const Line& L = city_.lines[e.line];
                nsm = 2 + e.line;
                boarding = (sm != nsm);
                bool suspended = false;
                double delay = cond ? cond->lineDelay(e.line, static_cast<int>(cur.t), &suspended) : 0.0;
                if (suspended) continue;
                double wait = boarding ? L.headway_min / 2.0 + delay : 0.0;
                double ride = e.dist_m / (L.speed_kmh * 1000.0 / 60.0);
                dt = wait + ride;
                dmoney = (boarding ? L.fare_base : 0.0) + L.fare_per_km * e.dist_m / 1000.0;
                dco2 = L.co2_g_pkm * e.dist_m / 1e6;
                crowd = ride * (city_.isPeak(static_cast<int>(cur.t)) ? 1.0 : 0.3);
            } else {  // CAR
                if (cond && (cond->car_closed_zones.count(city_.nodes[u].zone) || cond->car_closed_zones.count(city_.nodes[e.to].zone)))
                    continue;
                nsm = kCar;
                boarding = (sm != kCar);
                dt = (boarding ? city_.params.car_access_min : 0.0) + e.dist_m / city_.carSpeedMpm(static_cast<int>(cur.t), congestion);
                dmoney = (boarding ? city_.params.taxi_base : 0.0) + city_.params.taxi_per_km * e.dist_m / 1000.0;
                dco2 = city_.params.car_co2_g_pkm * e.dist_m / 1e6;
            }
            bool transfer = boarding && cur.boardings > 0;
            double dg = req.w.time * dt + req.w.money * dmoney + req.w.co2 * dco2 + req.w.dist * e.dist_m / 1000.0 +
                        (transfer ? req.w.transfer : 0.0) + req.w.transit_crowd * crowd;
            int ns = e.to * S + nsm;
            if (out.settled[ns]) continue;
            double ng = cur.g + dg;
            if (ng >= best[ns] - 1e-12) continue;
            best[ns] = ng;
            Label& nl = out.labels[ns];
            nl.g = ng;
            nl.t = cur.t + dt;
            nl.money = cur.money + dmoney;
            nl.co2 = cur.co2 + dco2;
            nl.walk = cur.walk + dwalk;
            nl.dist = cur.dist + e.dist_m;
            nl.crowd = cur.crowd + crowd;
            nl.boardings = cur.boardings + (boarding ? 1 : 0);
            nl.parent = s;
            nl.edge_mode = e.mode;
            nl.edge_line = e.line;
            nl.edge_dist = e.dist_m;
            nl.edge_depart = cur.t;
            double h = heuristic(e.to, goal, req);
            double f = alg == Algorithm::UCS ? ng : (alg == Algorithm::GREEDY ? h : ng + h);
            pq.push({f, ns});
            ++out.generated;
        }
    }
    return out;
}

RouteResult Router::extract(const SearchOutput& out, int goalNode, const RouteRequest& req) const {
    RouteResult r;
    r.expanded = out.expanded;
    r.generated = out.generated;
    r.depart = req.depart;
    const int S = nStateModes();
    if (goalNode == req.from) {
        r.found = true;
        r.arrive = req.depart;
        r.path = {req.from};
        return r;
    }
    int bestState = -1;
    double bestG = kInf;
    for (int m = 0; m < S; ++m) {
        int s = goalNode * S + m;
        if (out.settled[s] && out.labels[s].g < bestG) { bestG = out.labels[s].g; bestState = s; }
    }
    if (bestState < 0) return r;
    const Label& L = out.labels[bestState];
    r.found = true;
    r.cost = L.g;
    r.arrive = L.t;
    r.time_min = L.t - req.depart;
    r.money = L.money;
    r.co2_kg = L.co2;
    r.walk_m = L.walk;
    r.dist_m = L.dist;
    r.crowd_exposure = L.crowd;
    r.transfers = std::max(0, L.boardings - 1);

    std::vector<int> states;
    for (int s = bestState; s >= 0; s = out.labels[s].parent) states.push_back(s);
    std::reverse(states.begin(), states.end());
    for (int s : states) r.path.push_back(s / S);

    for (std::size_t i = 1; i < states.size(); ++i) {
        const Label& a = out.labels[states[i - 1]];
        const Label& b = out.labels[states[i]];
        int mode = b.edge_mode, line = b.edge_line;
        bool extend = !r.legs.empty() && r.legs.back().mode == mode && r.legs.back().line == line;
        if (!extend) {
            Leg leg;
            leg.mode = mode;
            leg.line = line;
            leg.from = states[i - 1] / S;
            leg.depart = b.edge_depart;
            r.legs.push_back(leg);
        }
        Leg& leg = r.legs.back();
        leg.to = states[i] / S;
        leg.arrive = b.t;
        leg.dist_m += b.edge_dist;
        leg.money += b.money - a.money;
        leg.co2_kg += b.co2 - a.co2;
    }
    return r;
}

RouteResult Router::route(const RouteRequest& req) const {
    auto t0 = std::chrono::steady_clock::now();
    SearchOutput out = run(req, nullptr, 1);
    RouteResult r = extract(out, req.to, req);
    r.runtime_us = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count();
    return r;
}

std::vector<RouteResult> Router::routeToMany(const RouteRequest& base, const std::vector<int>& targets) const {
    std::vector<char> mask(city_.nodes.size(), 0);
    int n = 0;
    for (int t : targets) if (!mask[t]) { mask[t] = 1; ++n; }
    SearchOutput out = run(base, &mask, n);
    std::vector<RouteResult> res;
    res.reserve(targets.size());
    for (int t : targets) res.push_back(extract(out, t, base));
    return res;
}

Json routeToJson(const RouteResult& r, const City& city, bool withPath) {
    Json j = Json::object();
    j["found"] = r.found;
    if (!r.found) return j;
    j["cost"] = r.cost;
    j["depart"] = r.depart;
    j["arrive"] = r.arrive;
    j["time_min"] = r.time_min;
    j["money"] = r.money;
    j["co2_kg"] = r.co2_kg;
    j["walk_m"] = r.walk_m;
    j["dist_m"] = r.dist_m;
    j["transfers"] = r.transfers;
    j["crowd_exposure"] = r.crowd_exposure;
    j["expanded"] = static_cast<double>(r.expanded);
    j["generated"] = static_cast<double>(r.generated);
    j["runtime_us"] = r.runtime_us;
    Json legs = Json::array();
    for (const Leg& l : r.legs) {
        Json lj = Json::object();
        lj["mode"] = modeName(city, l.mode, l.line);
        if (l.line >= 0) lj["line"] = city.lines[l.line].id;
        lj["from"] = l.from;
        lj["to"] = l.to;
        lj["depart"] = l.depart;
        lj["arrive"] = l.arrive;
        lj["dist_m"] = l.dist_m;
        lj["money"] = l.money;
        lj["co2_kg"] = l.co2_kg;
        legs.push(lj);
    }
    j["legs"] = legs;
    if (withPath) {
        Json p = Json::array();
        for (int n : r.path) p.push(n);
        j["path"] = p;
    }
    return j;
}

}  // namespace tp
