#include "planner.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <limits>
#include <queue>
#include <unordered_map>
#include <functional>
#include <string>
#include <utility>
#include <vector>

namespace tp {

namespace {

// Resource model (per minute unless stated otherwise).
constexpr double kWalkEnergy = 0.25;
constexpr double kRideEnergy = 0.04;
constexpr double kVisitEnergy = 0.08;
constexpr double kHunger = 0.1;
constexpr double kMinEnergyVisit = 20;
constexpr double kMaxHungerVisit = 85;
constexpr double kRestTrigger = 40;
constexpr double kRestGain = 40;
constexpr int kRestMin = 30;
constexpr int kTicketMin = 5;
constexpr double kLunchPenalty = 3.0;
constexpr double kCrowdedMargin = 0.2;
constexpr double kTravelActionCost = 0.25;  // discourages pointless back-and-forth trips

struct Goal {
    int node = 0, poi = 0;
    double utility = 0, price = 0;
    int dur = 60;
    bool must = false, restaurant = false, local = false;
    int modes = -1;
    int replaces = -1;  // for alternatives: the node of the goal it can replace
};

struct State {
    int node = 0;
    double t = 0, budget = 0, energy = 100, hunger = 20, walked = 0;
    uint32_t done = 0, skipped = 0, shifted = 0;
    bool lunch = false;
    double g = 0;
    double util = 0, money = 0, co2 = 0, wait = 0, travel = 0, crowdpen = 0, local = 0;
    bool terminal = false;
};

struct Action {
    std::string type;
    int poi = -1, from = -1, to = -1, modes = 0;
    double start = 0, end = 0, wait = 0, rho = 0, rho_after = 0, money = 0, co2 = 0, walk = 0, utility = 0, dist = 0;
    bool car = false;
    std::string note;
};

struct SearchNode {
    State s;
    int parent = -1;
    std::vector<Action> acts;
};

class Planner {
public:
    Planner(const City& city, const Context* ctx, const Json& req)
        : city_(city), ctx_(ctx), prof_(Profile::fromJson(req["profile"])), w_(Weights::fromJson(req["weights"])),
          cond_(ctx ? &ctx->cond : nullptr), legs_(city, prof_, w_, cond_) {
        const Json& o = req["options"];
        maxExp_ = static_cast<long>(o.getNum("max_expansions", 30000));
        allowShift_ = o.getBool("allow_shift", true);
        shiftWindow_ = o.getInt("shift_window", 150);
        window_ = o.getInt("window", 3);
        vmax_ = 1e-9;
        if (prof_.modes & (1 << WALK)) vmax_ = std::max(vmax_, prof_.accessible_only ? city.params.wheel_mpm : city.params.walk_mpm);
        if (prof_.modes & (1 << TRANSIT)) vmax_ = std::max(vmax_, city.max_transit_mpm);
        if (prof_.modes & (1 << CAR)) vmax_ = std::max(vmax_, city.params.car_kmh_off * 1000.0 / 60.0);

        auto readGoal = [&](const Json& gj, Goal& g) {
            g.node = gj.getInt("poi", -1);
            auto it = city.poiIndex.find(g.node);
            if (it == city.poiIndex.end()) return false;
            g.poi = it->second;
            const Poi& p = city.pois[g.poi];
            g.utility = gj.getNum("utility", 1.0);
            g.price = gj.getNum("price", p.price);
            g.dur = gj.getInt("duration_min", static_cast<int>(p.duration_min));
            g.must = gj.getBool("must", false);
            g.restaurant = gj.getBool("restaurant", p.kind == "restaurant");
            g.local = gj.getBool("local", p.independent);
            g.modes = gj.has("modes") ? modesFromJson(gj["modes"], -1) : -1;
            g.replaces = gj.getInt("replaces", -1);
            return true;
        };
        for (const auto& gj : req["goals"].items()) {
            Goal g;
            if (readGoal(gj, g)) goals_.push_back(g);
        }
        nBase_ = static_cast<int>(goals_.size());
        for (const auto& gj : req["alternatives"].items()) {
            Goal g;
            if (readGoal(gj, g)) { g.must = false; goals_.push_back(g); }
        }
        if (goals_.size() > 30) goals_.resize(30);

        // hint order (CSP agenda) for the base goals
        for (const auto& n : req["hint_order"].items()) {
            for (int i = 0; i < nBase_; ++i) if (goals_[i].node == n.integer()) hint_.push_back(i);
        }
        if (hint_.empty()) for (int i = 0; i < nBase_; ++i) hint_.push_back(i);
        for (int i = 0; i < nBase_; ++i)
            if (std::find(hint_.begin(), hint_.end(), i) == hint_.end()) hint_.push_back(i);
        for (const auto& n : req["previous_order"].items()) {
            for (int i = 0; i < nBase_; ++i) if (goals_[i].node == n.integer()) prev_.push_back(i);
        }

        const Json& st = req["state"];
        start_.node = st.getInt("node", prof_.start_node);
        start_.t = st.getNum("time", prof_.start_min);
        start_.budget = st.getNum("budget", prof_.budget);
        start_.energy = st.getNum("energy", 100);
        start_.hunger = st.getNum("hunger", 20);
        start_.walked = st.getNum("walked_m", 0);
        start_.lunch = st.getBool("lunch_done", false);

        std::vector<int> targets{prof_.start_node, prof_.end_node, start_.node};
        for (const auto& g : goals_) targets.push_back(g.node);
        legs_.addTargets(targets);
    }

    Json plan();
    Json replan();

private:
    struct AgendaResult {
        std::vector<Action> acts;
        State end;
        std::vector<std::pair<int, std::string>> failures;
        int crowded = 0;
        double cost = 0;
        int conflicts() const { return static_cast<int>(failures.size()) + crowded; }
    };

    int modesFor(int gi) const { return goals_[gi].modes > 0 ? goals_[gi].modes : prof_.modes; }
    const Poi& poiOf(int gi) const { return city_.pois[goals_[gi].poi]; }
    double h(const State& s) const { return w_.time * city_.euclid(s.node, prof_.end_node) / vmax_; }
    double skipCost(int gi) const {
        const Goal& g = goals_[gi];
        if (gi >= nBase_) return 0.0;  // alternatives are optional
        return w_.utility * g.utility + (g.local ? w_.local : 0.0) + (g.restaurant && prof_.lunch ? kLunchPenalty : 0.0);
    }
    bool resolved(const State& s, int gi) const { return ((s.done | s.skipped) >> gi) & 1u; }
    bool allResolved(const State& s) const {
        for (int i = 0; i < static_cast<int>(goals_.size()); ++i) if (!resolved(s, i)) return false;
        return true;
    }

    bool travel(const State& s, int to, int modes, State& ns, Action& a, std::string* why);
    bool visit(const State& s, int gi, State& ns, std::vector<Action>& acts, std::string* why);
    bool shift(const State& s, int gi, State& ns, Action& a);
    void rest(const State& s, State& ns, Action& a) const;
    bool finish(const State& s, State& ns, std::vector<Action>& acts, std::string* why);

    bool astar(const State& start, std::vector<Action>& out, State& end, long& expanded, long& generated);
    AgendaResult follow(const std::vector<int>& order, const std::vector<char>& shiftAllowed, const State& start);
    double agendaObjective(const AgendaResult& r) const;
    Json output(const std::vector<Action>& acts, const State& end, const std::string& strategy);

    const City& city_;
    const Context* ctx_;
    Profile prof_;
    Weights w_;
    const Conditions* cond_;
    LegRouter legs_;
    std::vector<Goal> goals_;
    int nBase_ = 0;
    std::vector<int> hint_, prev_;
    State start_;
    long maxExp_ = 30000;
    bool allowShift_ = true;
    int shiftWindow_ = 150, window_ = 3;
    double vmax_ = 1;
    std::chrono::steady_clock::time_point t0_ = std::chrono::steady_clock::now();
};

bool Planner::travel(const State& s, int to, int modes, State& ns, Action& a, std::string* why) {
    const LegRouter::Leg& L = legs_.leg(s.node, to, s.t, modes);
    if (!L.found) { if (why) *why = "no route available"; return false; }
    if (s.walked + L.walk > prof_.max_walk_m + 1e-6) { if (why) *why = "walking limit reached"; return false; }
    if (s.budget + 1e-9 < L.money) { if (why) *why = "budget exhausted"; return false; }
    ns = s;
    double walkMin = L.walk / (prof_.accessible_only ? city_.params.wheel_mpm : city_.params.walk_mpm);
    double ride = std::max(0.0, L.time - walkMin);
    ns.node = to;
    ns.t = s.t + L.time;
    ns.energy = s.energy - walkMin * kWalkEnergy - ride * kRideEnergy;
    ns.hunger = s.hunger + L.time * kHunger;
    ns.budget = s.budget - L.money;
    ns.walked = s.walked + L.walk;
    ns.money += L.money;
    ns.co2 += L.co2;
    ns.travel += L.time;
    ns.g += L.cost + kTravelActionCost;
    if (ns.t > prof_.end_min + 1e-6) { if (why) *why = "would end after the day window"; return false; }
    a = Action();
    a.type = "travel";
    a.from = s.node;
    a.to = to;
    a.start = s.t;
    a.end = ns.t;
    a.money = L.money;
    a.co2 = L.co2;
    a.walk = L.walk;
    a.dist = L.dist;
    a.car = L.car;
    a.modes = modes;
    return true;
}

bool Planner::visit(const State& s, int gi, State& ns, std::vector<Action>& acts, std::string* why) {
    const Goal& g = goals_[gi];
    const Poi& p = poiOf(gi);
    if (s.node != g.node) { if (why) *why = "not at the location"; return false; }
    if (cond_ && cond_->closed_pois.count(g.node)) { if (why) *why = "closed today"; return false; }
    double t = s.t;
    if (t + 1e-6 < p.open_min) { if (why) *why = "not open yet"; return false; }
    if (!g.restaurant) {
        if (s.energy < kMinEnergyVisit) { if (why) *why = "too tired, needs rest"; return false; }
        if (s.hunger >= kMaxHungerVisit) { if (why) *why = "needs a meal first"; return false; }
    } else if (prof_.lunch && (t < prof_.lunch_from - 30 || t > prof_.lunch_to + 30)) {
        if (why) *why = "outside the meal window";
        return false;
    }
    for (int j = 0; j < static_cast<int>(goals_.size()); ++j) {
        if (j != gi && goals_[j].node == g.node && ((s.done >> j) & 1u)) {
            if (why) *why = "already visited";
            return false;
        }
    }
    if (s.budget + 1e-9 < g.price) { if (why) *why = "budget exhausted"; return false; }
    CrowdEstimate ce = estimateCrowd(city_, ctx_, g.poi, static_cast<int>(t), static_cast<int>(t) + g.dur,
                                     prof_.crowd_threshold, w_.planned_load);
    if (ce.rho_max > prof_.hard_capacity) { if (why) *why = "at capacity"; return false; }
    double ticket = (p.ticketed && g.price > 0) ? kTicketMin : 0;
    double begin = t + ticket + ce.wait;
    double end = begin + g.dur;
    if (end > p.close_min + 1e-6) { if (why) *why = "closes before the visit would end"; return false; }
    if (end > prof_.end_min + 1e-6) { if (why) *why = "would end after the day window"; return false; }
    ns = s;
    ns.t = end;
    ns.budget -= g.price;
    ns.money += g.price;
    ns.wait += ce.wait;
    ns.crowdpen += ce.penalty;
    if (g.restaurant) {
        ns.energy = std::min(100.0, s.energy + 15);
        ns.hunger = 0;
        ns.lunch = true;
    } else {
        ns.energy = s.energy - g.dur * kVisitEnergy;
        ns.hunger = s.hunger + (end - t) * kHunger;
    }
    ns.done |= (1u << gi);
    ns.util += g.utility;
    // a goal and its alternatives are mutually exclusive: doing one resolves the others
    const int nGoals = static_cast<int>(goals_.size());
    if (gi >= nBase_) {
        for (int b = 0; b < nBase_; ++b) {
            if (goals_[b].node != g.replaces || resolved(ns, b)) continue;
            ns.skipped |= (1u << b);
            ns.g += std::max(0.0, skipCost(b) - w_.utility * g.utility);
        }
    } else {
        for (int a = nBase_; a < nGoals; ++a) if (goals_[a].replaces == g.node) ns.skipped |= (1u << a);
    }
    if (g.local) ns.local += 1;
    ns.g += w_.wait * ce.wait + w_.crowd * ce.penalty;
    if (ticket > 0) {
        Action b;
        b.type = "buy_ticket";
        b.poi = g.node;
        b.start = t;
        b.end = t + ticket;
        b.money = g.price;
        acts.push_back(b);
    }
    Action a;
    a.type = g.restaurant ? "eat" : ((p.kind == "attraction") ? "visit" : "local_experience");
    a.poi = g.node;
    a.start = t + ticket;
    a.wait = ce.wait;
    a.end = end;
    a.rho = ce.rho_max;
    a.utility = g.utility;
    a.money = ticket > 0 ? 0 : g.price;
    acts.push_back(a);
    return true;
}

bool Planner::shift(const State& s, int gi, State& ns, Action& a) {
    const Goal& g = goals_[gi];
    const Poi& p = poiOf(gi);
    if (s.node != g.node || ((s.shifted >> gi) & 1u)) return false;
    if (cond_ && cond_->closed_pois.count(g.node)) return false;
    const int slot = city_.slot_min;
    double t = s.t;
    bool beforeOpen = t < p.open_min;
    double curCost = std::numeric_limits<double>::infinity();
    double curRho = 0, curWait = 0;
    if (!beforeOpen) {
        if (!allowShift_) return false;
        CrowdEstimate cur = estimateCrowd(city_, ctx_, g.poi, static_cast<int>(t), static_cast<int>(t) + g.dur,
                                          prof_.crowd_threshold, w_.planned_load);
        curCost = w_.crowd * cur.penalty + w_.wait * cur.wait;
        curRho = cur.rho_max;
        curWait = cur.wait;
        if (cur.rho_max <= prof_.crowd_threshold && cur.rho_max <= prof_.hard_capacity) return false;
    }
    int first = city_.day_start + static_cast<int>(std::ceil((std::max(t, static_cast<double>(p.open_min)) - city_.day_start) / slot)) * slot;
    if (first <= t) first += slot;
    double bestT = -1, bestCost = curCost, bestRho = 0;
    for (int t2 = beforeOpen ? std::max(first - slot, p.open_min) : first; t2 <= t + shiftWindow_; t2 += slot) {
        if (t2 < p.open_min) continue;
        if (t2 + g.dur > p.close_min || t2 + g.dur > prof_.end_min) break;
        CrowdEstimate ce = estimateCrowd(city_, ctx_, g.poi, t2, t2 + g.dur, prof_.crowd_threshold, w_.planned_load);
        if (ce.rho_max > prof_.hard_capacity) continue;
        double c = w_.crowd * ce.penalty + w_.wait * ce.wait + w_.time * (t2 - t) * 0.5;
        if (beforeOpen && bestT < 0) { bestT = t2; bestCost = c; bestRho = ce.rho_max; continue; }
        // only worth waiting for a clearly quieter slot (or a much shorter queue)
        bool meaningful = beforeOpen || ce.rho_max <= curRho - 0.1 || ce.wait <= curWait - 5;
        if (meaningful && c < bestCost - 0.05) { bestT = t2; bestCost = c; bestRho = ce.rho_max; }
    }
    if (bestT < 0) return false;
    ns = s;
    double dt = bestT - t;
    ns.t = bestT;
    ns.shifted |= (1u << gi);
    ns.energy = std::min(100.0, s.energy + dt * 0.1);
    ns.hunger = s.hunger + dt * kHunger * 0.5;
    ns.g += w_.time * dt * 0.5;
    a = Action();
    a.type = beforeOpen ? "wait" : "shift_visit";
    a.poi = g.node;
    a.start = t;
    a.end = bestT;
    a.rho = curRho;
    a.rho_after = bestRho;
    a.note = beforeOpen ? "wait for opening" : "move to a quieter time slot";
    return true;
}

void Planner::rest(const State& s, State& ns, Action& a) const {
    ns = s;
    ns.t = s.t + kRestMin;
    ns.energy = std::min(100.0, s.energy + kRestGain);
    ns.hunger = s.hunger + kRestMin * kHunger * 0.5;
    ns.g += w_.time * kRestMin * 0.5;
    a = Action();
    a.type = "rest";
    a.poi = s.node;
    a.start = s.t;
    a.end = ns.t;
}

bool Planner::finish(const State& s, State& ns, std::vector<Action>& acts, std::string* why) {
    if (s.node == prof_.end_node) {
        ns = s;
        ns.terminal = true;
        return true;
    }
    Action a;
    if (!travel(s, prof_.end_node, prof_.modes, ns, a, why)) return false;
    ns.terminal = true;
    acts.push_back(a);
    return true;
}

bool Planner::astar(const State& start, std::vector<Action>& out, State& end, long& expanded, long& generated) {
    std::vector<SearchNode> arena;
    arena.reserve(4096);
    using QItem = std::pair<double, int>;
    std::priority_queue<QItem, std::vector<QItem>, std::greater<QItem>> pq;
    std::unordered_map<unsigned long long, double> closed;
    arena.push_back({start, -1, {}});
    pq.push({h(start), 0});
    generated = 1;
    expanded = 0;
    const int nGoals = static_cast<int>(goals_.size());

    auto keyOf = [](const State& s) {
        unsigned long long k = static_cast<unsigned long long>(s.node);
        k = k * 1000003ULL + s.done;
        k = k * 1000003ULL + s.skipped;
        k = k * 1000003ULL + s.shifted;
        k = k * 31ULL + (s.lunch ? 1 : 0);
        k = k * 1009ULL + static_cast<unsigned long long>(s.t / 5.0);
        k = k * 13ULL + static_cast<unsigned long long>(std::max(0.0, s.energy) / 10.0);
        return k;
    };
    auto push = [&](int parent, const State& ns, std::vector<Action> acts) {
        arena.push_back({ns, parent, std::move(acts)});
        pq.push({ns.g + (ns.terminal ? 0.0 : h(ns)), static_cast<int>(arena.size()) - 1});
        ++generated;
    };

    while (!pq.empty()) {
        int idx = pq.top().second;
        pq.pop();
        const State s = arena[idx].s;
        if (s.terminal) {
            std::vector<int> chain;
            for (int i = idx; i >= 0; i = arena[i].parent) chain.push_back(i);
            std::reverse(chain.begin(), chain.end());
            out.clear();
            for (int i : chain) for (const auto& a : arena[i].acts) out.push_back(a);
            end = s;
            return true;
        }
        unsigned long long k = keyOf(s);
        auto it = closed.find(k);
        if (it != closed.end() && it->second <= s.g + 1e-9) continue;
        closed[k] = s.g;
        if (++expanded > maxExp_) return false;

        if (allResolved(s)) {
            State ns;
            std::vector<Action> acts;
            if (finish(s, ns, acts, nullptr)) push(idx, ns, acts);
            continue;
        }
        if (s.energy < kRestTrigger) {
            State ns;
            Action a;
            rest(s, ns, a);
            push(idx, ns, {a});
        }
        // pending goals considered for travel: the first `window_` base goals in agenda order + alternatives
        std::vector<int> travelable;
        int taken = 0;
        for (int gi : hint_) {
            if (resolved(s, gi)) continue;
            if (taken < window_) travelable.push_back(gi);
            ++taken;
        }
        for (int gi = nBase_; gi < nGoals; ++gi) if (!resolved(s, gi)) travelable.push_back(gi);

        for (int gi = 0; gi < nGoals; ++gi) {
            if (resolved(s, gi)) continue;
            const Goal& g = goals_[gi];
            if (s.node == g.node) {
                State ns;
                std::vector<Action> acts;
                if (visit(s, gi, ns, acts, nullptr)) push(idx, ns, acts);
                Action a;
                if (shift(s, gi, ns, a)) push(idx, ns, {a});
            } else if (std::find(travelable.begin(), travelable.end(), gi) != travelable.end()) {
                State ns;
                Action a;
                if (travel(s, g.node, modesFor(gi), ns, a, nullptr) && ns.t + g.dur <= poiOf(gi).close_min)
                    push(idx, ns, {a});
            }
            if (!g.must) {
                State ns = s;
                ns.skipped |= (1u << gi);
                ns.g += skipCost(gi);
                Action a;
                a.type = "skip";
                a.poi = g.node;
                a.start = a.end = s.t;
                push(idx, ns, gi >= nBase_ ? std::vector<Action>{} : std::vector<Action>{a});
            }
        }
    }
    return false;
}

Planner::AgendaResult Planner::follow(const std::vector<int>& order, const std::vector<char>& shiftAllowed,
                                      const State& start) {
    AgendaResult r;
    State s = start;
    for (int gi : order) {
        if (resolved(s, gi)) continue;
        const Goal& g = goals_[gi];
        std::string why;
        if (s.energy < kRestTrigger) {
            State ns;
            Action a;
            rest(s, ns, a);
            s = ns;
            r.acts.push_back(a);
        }
        if (s.node != g.node) {
            State ns;
            Action a;
            if (!travel(s, g.node, modesFor(gi), ns, a, &why)) {
                r.failures.emplace_back(g.node, why);
                s.skipped |= (1u << gi);
                s.g += skipCost(gi);
                continue;
            }
            s = ns;
            r.acts.push_back(a);
        }
        const Poi& p = poiOf(gi);
        bool allow = gi < static_cast<int>(shiftAllowed.size()) && shiftAllowed[gi];
        if (s.t < p.open_min || allow) {
            bool saved = allowShift_;
            allowShift_ = allow;
            State ns;
            Action a;
            if (shift(s, gi, ns, a)) { s = ns; r.acts.push_back(a); }
            allowShift_ = saved;
        }
        State ns;
        std::vector<Action> acts;
        if (!visit(s, gi, ns, acts, &why)) {
            r.failures.emplace_back(g.node, why);
            s.skipped |= (1u << gi);
            s.g += skipCost(gi);
            continue;
        }
        for (const auto& a : acts) {
            if ((a.type == "visit" || a.type == "local_experience") && a.rho > prof_.crowd_threshold + kCrowdedMargin) ++r.crowded;
            r.acts.push_back(a);
        }
        s = ns;
    }
    // goals not in the agenda count as skipped
    for (int gi = 0; gi < nBase_; ++gi) if (!resolved(s, gi)) { s.skipped |= (1u << gi); s.g += skipCost(gi); }
    State ns;
    std::vector<Action> acts;
    std::string why;
    if (!finish(s, ns, acts, &why)) {
        r.failures.emplace_back(prof_.end_node, "cannot return to the end point in time");
        ns = s;
        ns.terminal = true;
    }
    for (const auto& a : acts) r.acts.push_back(a);
    r.end = ns;
    r.cost = ns.g;
    return r;
}

double Planner::agendaObjective(const AgendaResult& r) const {
    return r.cost + 1000.0 * r.conflicts();
}

Json Planner::output(const std::vector<Action>& acts, const State& end, const std::string& strategy) {
    Json out = Json::object();
    out["status"] = "ok";
    out["strategy"] = strategy;
    Json arr = Json::array();
    Json visited = Json::array();
    for (const Action& a : acts) {
        Json j = Json::object();
        j["type"] = a.type;
        j["start"] = a.start;
        j["end"] = a.end;
        if (a.type == "travel") {
            j["from"] = a.from;
            j["to"] = a.to;
            j["money"] = a.money;
            j["co2_kg"] = a.co2;
            j["walk_m"] = a.walk;
            j["dist_m"] = a.dist;
            j["car"] = a.car;
            RouteResult r = legs_.full(a.from, a.to, a.start, a.modes);
            Json segs = Json::array();
            for (const Leg& l : r.legs) {
                Json sj = Json::object();
                sj["mode"] = modeName(city_, l.mode, l.line);
                if (l.line >= 0) sj["line"] = city_.lines[l.line].id;
                sj["from"] = l.from;
                sj["to"] = l.to;
                sj["dist_m"] = l.dist_m;
                sj["co2_kg"] = l.co2_kg;
                segs.push(sj);
            }
            j["segments"] = segs;
        } else {
            j["poi"] = a.poi;
            if (const Poi* p = city_.poiAt(a.poi)) j["name"] = p->name;
        }
        if (a.type == "visit" || a.type == "eat" || a.type == "local_experience") {
            j["wait"] = a.wait;
            j["rho"] = a.rho;
            j["utility"] = a.utility;
            j["money"] = a.money;
            visited.push(a.poi);
        }
        if (a.type == "buy_ticket") j["money"] = a.money;
        if (a.type == "shift_visit" || a.type == "wait") {
            j["rho_before"] = a.rho;
            j["rho_after"] = a.rho_after;
        }
        if (!a.note.empty()) j["note"] = a.note;
        arr.push(j);
    }
    out["actions"] = arr;
    out["visited"] = visited;
    Json skipped = Json::array();
    for (int gi = 0; gi < nBase_; ++gi) if ((end.skipped >> gi) & 1u) skipped.push(goals_[gi].node);
    out["skipped"] = skipped;
    Json tot = Json::object();
    tot["utility"] = end.util;
    tot["money"] = end.money;
    tot["co2_kg"] = end.co2;
    tot["walk_m"] = end.walked;
    tot["travel_min"] = end.travel;
    tot["wait_min"] = end.wait;
    tot["end_time"] = end.t;
    tot["energy_end"] = end.energy;
    tot["local"] = end.local;
    tot["cost"] = end.g;
    out["totals"] = tot;
    return out;
}

Json Planner::plan() {
    std::vector<Action> acts;
    State end;
    long expanded = 0, generated = 0;
    Json out;
    if (astar(start_, acts, end, expanded, generated)) {
        out = output(acts, end, "astar");
    } else {
        // search budget exhausted: execute the CSP agenda directly
        std::vector<char> shiftAllowed(goals_.size(), allowShift_ ? 1 : 0);
        AgendaResult r = follow(hint_, shiftAllowed, start_);
        out = output(r.acts, r.end, "agenda");
        Json f = Json::array();
        for (const auto& x : r.failures) {
            Json fj = Json::object();
            fj["poi"] = x.first;
            fj["reason"] = x.second;
            f.push(fj);
        }
        out["failures"] = f;
    }
    Json st = Json::object();
    st["expanded"] = static_cast<double>(expanded);
    st["generated"] = static_cast<double>(generated);
    st["route_searches"] = static_cast<double>(legs_.searches());
    st["runtime_ms"] = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0_).count();
    out["stats"] = st;
    return out;
}

Json Planner::replan() {
    std::vector<int> order = prev_.empty() ? hint_ : prev_;
    std::vector<char> shiftAllowed(goals_.size(), 0);
    AgendaResult base = follow(order, shiftAllowed, start_);
    int conflictsBefore = base.conflicts();
    Json reasons = Json::array();
    for (const auto& f : base.failures) {
        Json fj = Json::object();
        fj["poi"] = f.first;
        fj["reason"] = f.second;
        reasons.push(fj);
    }
    auto stability = [&](const std::vector<Action>& acts) {
        std::vector<int> visitedNodes;
        for (const auto& a : acts)
            if (a.type == "visit" || a.type == "eat" || a.type == "local_experience") visitedNodes.push_back(a.poi);
        std::vector<int> prevNodes;
        for (int gi : order) prevNodes.push_back(goals_[gi].node);
        int changed = 0;
        std::size_t n = std::max(visitedNodes.size(), prevNodes.size());
        for (std::size_t i = 0; i < n; ++i) {
            int a = i < visitedNodes.size() ? visitedNodes[i] : -1;
            int b = i < prevNodes.size() ? prevNodes[i] : -1;
            if (a != b) ++changed;
        }
        return changed;
    };

    Json out;
    int iterations = 0;
    if (conflictsBefore == 0) {
        out = output(base.acts, base.end, "keep");
    } else {
        // Min-conflicts local repair: repeatedly fix one conflicted goal with the
        // value (shift / move / substitute / skip) that leaves the fewest conflicts.
        std::vector<int> cur = order;
        std::vector<char> curShift = shiftAllowed;
        AgendaResult curRes = base;
        for (iterations = 0; iterations < 25 && curRes.conflicts() > 0; ++iterations) {
            int gi = -1;
            for (const auto& f : curRes.failures) {
                for (int k : cur) if (goals_[k].node == f.first) { gi = k; break; }
                if (gi >= 0) break;
            }
            if (gi < 0) {
                for (const auto& a : curRes.acts) {
                    if ((a.type == "visit" || a.type == "local_experience") && a.rho > prof_.crowd_threshold + kCrowdedMargin)
                        for (int k : cur) if (goals_[k].node == a.poi) { gi = k; break; }
                    if (gi >= 0) break;
                }
            }
            if (gi < 0) break;
            struct Option { std::vector<int> order; std::vector<char> shift; };
            std::vector<Option> options;
            {   // allow shifting to a quieter slot
                Option o{cur, curShift};
                if (!o.shift[gi]) { o.shift[gi] = 1; options.push_back(o); }
            }
            {   // move later in the day
                Option o{cur, curShift};
                o.order.erase(std::find(o.order.begin(), o.order.end(), gi));
                o.order.push_back(gi);
                if (o.order != cur) options.push_back(o);
            }
            {   // move earlier
                Option o{cur, curShift};
                o.order.erase(std::find(o.order.begin(), o.order.end(), gi));
                o.order.insert(o.order.begin(), gi);
                if (o.order != cur) options.push_back(o);
            }
            for (int ai = nBase_; ai < static_cast<int>(goals_.size()); ++ai) {   // substitute
                if (goals_[ai].replaces != goals_[gi].node) continue;
                if (std::find(cur.begin(), cur.end(), ai) != cur.end()) continue;
                Option o{cur, curShift};
                *std::find(o.order.begin(), o.order.end(), gi) = ai;
                options.push_back(o);
            }
            if (!goals_[gi].must) {   // skip
                Option o{cur, curShift};
                o.order.erase(std::find(o.order.begin(), o.order.end(), gi));
                options.push_back(o);
            }
            double bestObj = agendaObjective(curRes);
            int bestK = -1;
            AgendaResult bestRes;
            for (std::size_t k = 0; k < options.size(); ++k) {
                AgendaResult r = follow(options[k].order, options[k].shift, start_);
                double obj = agendaObjective(r);
                if (obj < bestObj - 1e-9) { bestObj = obj; bestK = static_cast<int>(k); bestRes = r; }
            }
            if (bestK < 0) break;
            cur = options[bestK].order;
            curShift = options[bestK].shift;
            curRes = bestRes;
        }
        if (curRes.conflicts() == 0) {
            out = output(curRes.acts, curRes.end, "local_repair");
        } else {
            std::vector<Action> acts;
            State end;
            long expanded = 0, generated = 0;
            if (astar(start_, acts, end, expanded, generated)) {
                out = output(acts, end, "full_replan");
            } else {
                out = output(curRes.acts, curRes.end, "best_effort");
            }
            Json st = Json::object();
            st["expanded"] = static_cast<double>(expanded);
            st["generated"] = static_cast<double>(generated);
            out["search"] = st;
        }
    }
    std::vector<Action> finalActs;
    for (const auto& a : out["actions"].items()) {
        Action x;
        x.type = a.getStr("type", "");
        x.poi = a.getInt("poi", -1);
        finalActs.push_back(x);
    }
    out["stability_changes"] = stability(finalActs);
    out["conflicts_before"] = conflictsBefore;
    out["conflict_reasons"] = reasons;
    out["repair_iterations"] = iterations;
    out["runtime_ms"] = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0_).count();
    return out;
}

}  // namespace

Json planItinerary(const City& city, const Context* ctx, const Json& request) {
    Planner p(city, ctx, request);
    return p.plan();
}

Json replanItinerary(const City& city, const Context* ctx, const Json& request) {
    Planner p(city, ctx, request);
    return p.replan();
}

}  // namespace tp
