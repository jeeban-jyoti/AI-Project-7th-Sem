#include "csp.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <deque>
#include <limits>
#include <numeric>
#include <set>
#include <cstdlib>
#include <functional>
#include <string>
#include <utility>
#include <vector>

namespace tp {

namespace {
constexpr int SKIP = -1;
constexpr double kNegInf = -std::numeric_limits<double>::infinity();
}  // namespace

int modesFromJson(const Json& arr, int def) {
    if (!arr.isArray()) return def;
    int m = 0;
    for (const auto& v : arr.items()) {
        const std::string& s = v.str();
        if (s == "walk") m |= 1 << WALK;
        else if (s == "transit") m |= 1 << TRANSIT;
        else if (s == "car") m |= 1 << CAR;
    }
    return m ? (m | (1 << WALK)) : def;  // walking is always possible between modes
}

Profile Profile::fromJson(const Json& j) {
    Profile p;
    p.start_node = j.getInt("start_node", 0);
    p.end_node = j.getInt("end_node", p.start_node);
    p.start_min = j.getInt("start_min", p.start_min);
    p.end_min = j.getInt("end_min", p.end_min);
    p.budget = j.getNum("budget", p.budget);
    p.max_walk_m = j.getNum("max_walk_m", p.max_walk_m);
    p.accessible_only = j.getBool("accessible_only", false);
    p.modes = modesFromJson(j["modes"], kAllModes);
    p.max_activities = j.getInt("max_activities", p.max_activities);
    p.lunch = j.getBool("lunch", p.lunch);
    if (j["lunch_window"].isArray()) {
        p.lunch_from = j["lunch_window"].at(0).integer(p.lunch_from);
        p.lunch_to = j["lunch_window"].at(1).integer(p.lunch_to);
    }
    p.crowd_threshold = j.getNum("crowd_threshold", p.crowd_threshold);
    p.co2_target_kg = j.getNum("co2_target_kg", p.co2_target_kg);
    p.hard_capacity = j.getNum("hard_capacity", p.hard_capacity);
    p.max_transfers = j.getInt("max_transfers", p.max_transfers);
    return p;
}

RouteRequest LegRouter::makeReq(int from, int to, double depart, int modes) const {
    RouteRequest r;
    r.from = from;
    r.to = to;
    r.depart = static_cast<int>(depart);
    r.modes = modes;
    r.accessible_only = prof_.accessible_only;
    r.algorithm = Algorithm::ASTAR;
    r.w = w_;
    r.cond = cond_;
    return r;
}

void LegRouter::addTargets(const std::vector<int>& nodes) {
    if (isTarget_.empty()) isTarget_.assign(city_.nodes.size(), 0);
    for (int n : nodes) {
        if (n < 0 || n >= static_cast<int>(isTarget_.size()) || isTarget_[n]) continue;
        isTarget_[n] = 1;
        targets_.push_back(n);
    }
}

int LegRouter::regimeOf(double t) const {
    int r = city_.isPeak(static_cast<int>(t)) ? 1 : 0;
    if (cond_) {
        int bit = 1;
        for (const auto& d : cond_->delays) {
            if (bit >= 7) break;
            if (t >= d.from && t < d.to) r |= (1 << bit);
            ++bit;
        }
    }
    return r;
}

static LegRouter::Leg toLeg(const RouteResult& r) {
    LegRouter::Leg l;
    l.found = r.found;
    l.time = r.time_min;
    l.cost = r.cost;
    l.money = r.money;
    l.co2 = r.co2_kg;
    l.walk = r.walk_m;
    l.dist = r.dist_m;
    l.transfers = r.transfers;
    for (const auto& seg : r.legs) if (seg.mode == CAR) l.car = true;
    return l;
}

const LegRouter::Leg& LegRouter::leg(int from, int to, double depart, int modes) {
    int regime = regimeOf(depart);
    long long k = key(from, to, regime, modes);
    auto it = cache_.find(k);
    if (it != cache_.end()) return it->second;
    if (from == to) {
        Leg l;
        l.found = true;
        return cache_.emplace(k, l).first->second;
    }
    addTargets({to});
    // One uniform-cost search from `from` settles every target at once.
    std::vector<RouteResult> res = router_.routeToMany(makeReq(from, to, depart, modes), targets_);
    ++searches_;
    for (std::size_t i = 0; i < targets_.size(); ++i) {
        Leg l = targets_[i] == from ? Leg() : toLeg(res[i]);
        if (targets_[i] == from) l.found = true;
        cache_[key(from, targets_[i], regime, modes)] = l;
    }
    return cache_[k];
}

RouteResult LegRouter::full(int from, int to, double depart, int modes) const {
    if (from == to) {
        RouteResult r;
        r.found = true;
        r.depart = r.arrive = depart;
        r.path = {from};
        return r;
    }
    return router_.route(makeReq(from, to, depart, modes));
}

namespace {

class CspSolver {
public:
    CspSolver(const City& city, const Context* ctx, const Profile& prof, const Weights& w,
              std::vector<Candidate> cands, const CspOptions& opt)
        : city_(city), ctx_(ctx), prof_(prof), w_(w), cands_(std::move(cands)), opt_(opt),
          cond_(ctx ? &ctx->cond : nullptr), legs_(city, prof_, w, cond_) {
        n_ = static_cast<int>(cands_.size());
        nSlots_ = city.nSlots();
        vmax_ = 1e-9;
        if (prof_.modes & (1 << WALK)) vmax_ = std::max(vmax_, prof_.accessible_only ? city.params.wheel_mpm : city.params.walk_mpm);
        if (prof_.modes & (1 << TRANSIT)) vmax_ = std::max(vmax_, city.max_transit_mpm);
        if (prof_.modes & (1 << CAR)) vmax_ = std::max(vmax_, city.params.car_kmh_off * 1000.0 / 60.0);
        std::vector<int> targets{prof_.start_node, prof_.end_node};
        for (const auto& c : cands_) targets.push_back(c.node);
        legs_.addTargets(targets);
    }

    Json run();

private:
    struct Item { int c; int start; };
    struct Eval {
        bool ok = false;
        std::string why;
        double score = 0, utility = 0, penalty = 0, legcost = 0, money = 0, co2 = 0, walk = 0, travel = 0, wait = 0;
        int transfers = 0;
    };

    int modesFor(int c) const { return cands_[c].modes_override > 0 ? cands_[c].modes_override : prof_.modes; }
    double lb(int a, int b) const { return city_.euclid(a, b) / vmax_; }
    int slotIndex(int t) const { return (t - city_.day_start) / city_.slot_min; }
    bool isActivity(int c) const { return !cands_[c].restaurant; }
    double gainOf(int c) const { return w_.utility * cands_[c].utility + (cands_[c].local ? w_.local : 0.0) - w_.money * cands_[c].price; }

    void buildDomains();
    int ac3();
    Eval evaluate(const std::vector<Item>& tl);
    bool insertSorted(std::vector<Item>& tl, int c, int t) const;
    double upperBound(double partial) const;
    bool lunchPossible() const;
    void dfs();
    bool greedy();
    void solveOnce();
    Json buildOutput(const char* status);
    Json conflicts();

    const City& city_;
    const Context* ctx_;
    Profile prof_;
    Weights w_;
    std::vector<Candidate> cands_;
    CspOptions opt_;
    const Conditions* cond_;
    LegRouter legs_;
    int n_ = 0, nSlots_ = 0;
    double vmax_ = 1;

    std::vector<std::vector<int>> dom_;          // current domains (start minutes)
    std::vector<std::vector<double>> vcost_;     // [cand][slot] crowd+wait cost (NaN = invalid)
    std::vector<double> vwait_;                  // [cand*nSlots+slot] expected wait
    std::vector<std::string> unaryReason_;
    std::vector<int> assign_;                    // -2 unassigned, SKIP, or start
    std::vector<Item> tl_;
    Eval curEval_;
    int includedActs_ = 0;
    bool lunchDone_ = false;

    std::vector<Item> best_;
    double bestScore_ = kNegInf;
    bool haveBest_ = false;
    bool greedyFound_ = false;
    double greedyScore_ = 0;

    long nodes_ = 0, backtracks_ = 0, pruned_ = 0, fcRemoved_ = 0;
    int ac3Removed_ = 0;
    bool stop_ = false;
    std::chrono::steady_clock::time_point t0_;
    std::vector<std::string> relaxed_;
};

void CspSolver::buildDomains() {
    dom_.assign(n_, {});
    vcost_.assign(n_, std::vector<double>(nSlots_, std::numeric_limits<double>::quiet_NaN()));
    vwait_.assign(static_cast<std::size_t>(n_) * nSlots_, 0.0);
    unaryReason_.assign(n_, "");
    const int slot = city_.slot_min;
    for (int i = 0; i < n_; ++i) {
        const Candidate& c = cands_[i];
        const Poi& poi = city_.pois[c.poi];
        if (cond_ && cond_->closed_pois.count(poi.node)) { unaryReason_[i] = "closed today"; continue; }
        if (prof_.accessible_only && !poi.accessible) { unaryReason_[i] = "not wheelchair accessible"; continue; }
        double lo = std::max<double>(poi.open_min, prof_.start_min + lb(prof_.start_node, c.node));
        double hi = std::min<double>(poi.close_min, prof_.end_min - lb(c.node, prof_.end_node)) - c.dur;
        int t = city_.day_start + static_cast<int>(std::ceil((lo - city_.day_start) / slot)) * slot;
        int capacityBlocked = 0, crowdBlocked = 0;
        for (; t <= hi && t < city_.day_end; t += slot) {
            if (c.restaurant && prof_.lunch && (t < prof_.lunch_from || t > prof_.lunch_to)) continue;
            bool forbidden = false;
            for (const auto& f : c.forbid) if (t >= f.first && t < f.second) forbidden = true;
            if (forbidden) continue;
            CrowdEstimate ce = estimateCrowd(city_, ctx_, c.poi, t, t + c.dur, prof_.crowd_threshold, w_.planned_load);
            if (ce.rho_max > prof_.hard_capacity) { ++capacityBlocked; continue; }
            if (opt_.soft_as_hard_crowd && ce.rho_max > prof_.crowd_threshold) { ++crowdBlocked; continue; }
            if (t + ce.wait + c.dur > poi.close_min) continue;
            int s = slotIndex(t);
            vcost_[i][s] = w_.crowd * ce.penalty + w_.wait * ce.wait;
            vwait_[static_cast<std::size_t>(i) * nSlots_ + s] = ce.wait;
            dom_[i].push_back(t);
        }
        if (dom_[i].empty()) {
            if (capacityBlocked) unaryReason_[i] = "at capacity for every reachable time slot";
            else if (crowdBlocked) unaryReason_[i] = "above the crowd threshold at every reachable time";
            else if (c.restaurant && prof_.lunch) unaryReason_[i] = "cannot be reached within the lunch window";
            else unaryReason_[i] = "opening hours do not fit the available time";
        }
    }
}

// AC-3 over the binary no-overlap constraints between must-visit activities
// (pairs involving optional activities are always supported by SKIP).
int CspSolver::ac3() {
    std::vector<int> must;
    for (int i = 0; i < n_; ++i) if (cands_[i].must) must.push_back(i);
    auto compatible = [&](int i, int vi, int j, int vj) {
        double ei = vi + cands_[i].dur, ej = vj + cands_[j].dur;
        return ei + lb(cands_[i].node, cands_[j].node) <= vj || ej + lb(cands_[j].node, cands_[i].node) <= vi;
    };
    std::deque<std::pair<int, int>> q;
    for (int a : must) for (int b : must) if (a != b) q.emplace_back(a, b);
    int removed = 0;
    while (!q.empty()) {
        auto arc = q.front();
        q.pop_front();
        int i = arc.first, j = arc.second;
        if (dom_[j].empty() || dom_[i].empty()) continue;  // already infeasible; reported on its own
        std::vector<int> keep;
        for (int vi : dom_[i]) {
            bool ok = false;
            for (int vj : dom_[j]) if (compatible(i, vi, j, vj)) { ok = true; break; }
            if (ok) keep.push_back(vi);
        }
        if (keep.size() != dom_[i].size()) {
            removed += static_cast<int>(dom_[i].size() - keep.size());
            dom_[i] = keep;
            for (int k : must) if (k != i && k != j) q.emplace_back(k, i);
        }
    }
    return removed;
}

bool CspSolver::insertSorted(std::vector<Item>& tl, int c, int t) const {
    auto pos = std::find_if(tl.begin(), tl.end(), [&](const Item& it) { return it.start > t; });
    tl.insert(pos, Item{c, t});
    return true;
}

CspSolver::Eval CspSolver::evaluate(const std::vector<Item>& tl) {
    Eval e;
    int curNode = prof_.start_node;
    double curT = prof_.start_min;
    for (const Item& it : tl) {
        const Candidate& c = cands_[it.c];
        const Poi& poi = city_.pois[c.poi];
        const LegRouter::Leg& L = legs_.leg(curNode, c.node, curT, modesFor(it.c));
        if (!L.found) { e.why = "no route to " + poi.name; return e; }
        if (curT + L.time > it.start + 1e-6) { e.why = "cannot reach " + poi.name + " in time"; return e; }
        int s = slotIndex(it.start);
        double wait = vwait_[static_cast<std::size_t>(it.c) * nSlots_ + s];
        double end = it.start + wait + c.dur;
        if (end > poi.close_min + 1e-6) { e.why = poi.name + " closes before the visit ends"; return e; }
        e.money += L.money + c.price;
        e.co2 += L.co2;
        e.walk += L.walk;
        e.travel += L.time;
        e.transfers += L.transfers;
        e.wait += wait;
        e.legcost += L.cost + w_.money * c.price;
        e.penalty += vcost_[it.c][s];
        e.utility += w_.utility * c.utility + (c.local ? w_.local : 0.0);
        curNode = c.node;
        curT = end;
    }
    const LegRouter::Leg& back = legs_.leg(curNode, prof_.end_node, curT, prof_.modes);
    if (!back.found) { e.why = "no route back"; return e; }
    if (curT + back.time > prof_.end_min + 1e-6) { e.why = "cannot return before the end of the day"; return e; }
    e.money += back.money;
    e.co2 += back.co2;
    e.walk += back.walk;
    e.travel += back.time;
    e.transfers += back.transfers;
    e.legcost += back.cost;
    if (e.money > prof_.budget + 1e-6) { e.why = "over budget"; return e; }
    if (e.walk > prof_.max_walk_m + 1e-6) { e.why = "exceeds maximum walking distance"; return e; }
    if (opt_.soft_as_hard_co2 && e.co2 > prof_.co2_target_kg) { e.why = "exceeds CO2 target"; return e; }
    e.ok = true;
    e.score = e.utility - e.penalty - e.legcost;
    return e;
}

bool CspSolver::lunchPossible() const {
    if (!prof_.lunch || lunchDone_) return true;
    bool anyRestaurant = false;
    for (int i = 0; i < n_; ++i) {
        if (!cands_[i].restaurant) continue;
        anyRestaurant = true;
        if (assign_[i] == -2 && !dom_[i].empty()) return true;
    }
    return !anyRestaurant;  // no restaurant candidates at all: lunch handled outside the CSP
}

double CspSolver::upperBound(double partial) const {
    std::vector<double> gains;
    double mustGain = 0, bestRestaurant = 0;
    for (int i = 0; i < n_; ++i) {
        if (assign_[i] != -2 || dom_[i].empty()) continue;
        double minCost = std::numeric_limits<double>::infinity();
        for (int t : dom_[i]) minCost = std::min(minCost, vcost_[i][slotIndex(t)]);
        double g = gainOf(i) - minCost;
        if (cands_[i].must) mustGain += g;
        else if (cands_[i].restaurant) { if (!lunchDone_) bestRestaurant = std::max(bestRestaurant, g); }
        else if (g > 0) gains.push_back(g);
    }
    std::sort(gains.rbegin(), gains.rend());
    int cap = prof_.max_activities - includedActs_;
    double add = mustGain + bestRestaurant;
    for (int k = 0; k < static_cast<int>(gains.size()) && k < cap; ++k) add += gains[k];
    return partial + add;
}

void CspSolver::dfs() {
    if (stop_) return;
    if ((++nodes_ & 255) == 0) {
        double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0_).count();
        if (ms > opt_.time_limit_ms || nodes_ > opt_.node_limit) { stop_ = true; return; }
    }
    if (upperBound(curEval_.score) <= bestScore_ + 1e-9) { ++pruned_; return; }

    // MRV with degree and utility tie-breaks; forced SKIPs are taken immediately.
    int var = -1;
    std::size_t bestSize = std::numeric_limits<std::size_t>::max();
    int bestDegree = -1;
    for (int i = 0; i < n_; ++i) {
        if (assign_[i] != -2) continue;
        if (dom_[i].empty()) { var = i; break; }
        std::size_t sz = dom_[i].size();
        int degree = 0;
        for (int j = 0; j < n_; ++j) {
            if (j == i || assign_[j] != -2 || dom_[j].empty()) continue;
            if (dom_[j].front() <= dom_[i].back() && dom_[i].front() <= dom_[j].back()) ++degree;
        }
        bool better = sz < bestSize || (sz == bestSize && degree > bestDegree) ||
                      (sz == bestSize && degree == bestDegree && var >= 0 && cands_[i].utility > cands_[var].utility);
        if (var < 0 || better) { var = i; bestSize = sz; bestDegree = degree; }
    }

    if (var < 0) {  // complete assignment
        if (prof_.lunch && !lunchDone_ && !lunchPossible()) return;
        bool hasRestaurant = false;
        for (int i = 0; i < n_; ++i) hasRestaurant |= cands_[i].restaurant;
        if (prof_.lunch && hasRestaurant && !lunchDone_) return;
        if (curEval_.ok && curEval_.score > bestScore_) {
            bestScore_ = curEval_.score;
            best_ = tl_;
            haveBest_ = true;
        }
        return;
    }

    const Candidate& c = cands_[var];
    std::vector<int> values;
    if (!dom_[var].empty()) {
        std::vector<int> sorted = dom_[var];
        std::stable_sort(sorted.begin(), sorted.end(), [&](int a, int b) {
            return vcost_[var][slotIndex(a)] < vcost_[var][slotIndex(b)];
        });
        int limit = c.must ? static_cast<int>(sorted.size()) : opt_.value_limit;
        for (int t : sorted) {
            if (static_cast<int>(values.size()) >= limit) break;
            bool close = false;
            for (int v : values) if (std::abs(v - t) < 2 * city_.slot_min) close = true;
            if (!close || c.must) values.push_back(t);
        }
        // least-constraining value: fewer eliminations in other domains first
        std::vector<std::pair<double, int>> scored;
        for (int t : values) {
            int elim = 0;
            for (int j = 0; j < n_; ++j) {
                if (j == var || assign_[j] != -2) continue;
                for (int v : dom_[j]) {
                    bool before = v + cands_[j].dur + lb(cands_[j].node, c.node) <= t;
                    bool after = v >= t + c.dur + lb(c.node, cands_[j].node);
                    if (!before && !after) ++elim;
                }
            }
            scored.emplace_back(vcost_[var][slotIndex(t)] + 0.02 * elim, t);
        }
        std::stable_sort(scored.begin(), scored.end());
        values.clear();
        for (auto& p : scored) values.push_back(p.second);
    }
    if (!c.must) values.push_back(SKIP);

    for (int v : values) {
        if (stop_) return;
        auto savedDom = dom_;
        Eval savedEval = curEval_;
        int savedActs = includedActs_;
        bool savedLunch = lunchDone_;
        assign_[var] = v;
        bool ok = true;
        if (v != SKIP) {
            if (isActivity(var) && includedActs_ >= prof_.max_activities) ok = false;
            if (ok) {
                insertSorted(tl_, var, v);
                Eval e = evaluate(tl_);
                if (!e.ok) ok = false;
                else {
                    curEval_ = e;
                    if (isActivity(var)) ++includedActs_;
                    if (c.restaurant) lunchDone_ = true;
                    double waitV = vwait_[static_cast<std::size_t>(var) * nSlots_ + slotIndex(v)];
                    double endV = v + waitV + c.dur;
                    for (int j = 0; j < n_ && ok; ++j) {
                        if (assign_[j] != -2) continue;
                        std::vector<int> keep;
                        bool clearAll = (cands_[j].restaurant && lunchDone_) ||
                                        (isActivity(j) && includedActs_ >= prof_.max_activities) ||
                                        (curEval_.money + cands_[j].price > prof_.budget);
                        if (!clearAll) {
                            for (int t : dom_[j]) {
                                bool before = t + cands_[j].dur + lb(cands_[j].node, c.node) <= v;
                                bool after = t >= endV + lb(c.node, cands_[j].node);
                                if (before || after) keep.push_back(t);
                            }
                        }
                        fcRemoved_ += static_cast<long>(dom_[j].size() - keep.size());
                        dom_[j] = keep;
                        if (cands_[j].must && dom_[j].empty()) ok = false;
                    }
                    if (ok && !lunchPossible()) ok = false;
                }
                if (!ok) {
                    auto it = std::find_if(tl_.begin(), tl_.end(), [&](const Item& x) { return x.c == var; });
                    if (it != tl_.end()) tl_.erase(it);
                }
            }
        } else {
            if (!lunchPossible()) ok = false;
        }
        if (ok) {
            dfs();
            if (v != SKIP) {
                auto it = std::find_if(tl_.begin(), tl_.end(), [&](const Item& x) { return x.c == var; });
                if (it != tl_.end()) tl_.erase(it);
            }
        } else {
            ++backtracks_;
        }
        assign_[var] = -2;
        dom_ = savedDom;
        curEval_ = savedEval;
        includedActs_ = savedActs;
        lunchDone_ = savedLunch;
    }
}

bool CspSolver::greedy() {
    std::vector<int> order(n_);
    std::iota(order.begin(), order.end(), 0);
    std::stable_sort(order.begin(), order.end(), [&](int a, int b) {
        if (cands_[a].must != cands_[b].must) return cands_[a].must;
        return gainOf(a) > gainOf(b);
    });
    std::vector<Item> tl;
    Eval cur = evaluate(tl);
    if (!cur.ok) return false;
    int acts = 0;
    bool lunch = false;
    for (int i : order) {
        const Candidate& c = cands_[i];
        if (c.restaurant && lunch) continue;
        if (isActivity(i) && acts >= prof_.max_activities && !c.must) continue;
        Eval bestE;
        int bestT = SKIP;
        for (int t : dom_[i]) {
            std::vector<Item> trial = tl;
            insertSorted(trial, i, t);
            Eval e = evaluate(trial);
            if (e.ok && (!bestE.ok || e.score > bestE.score)) { bestE = e; bestT = t; }
        }
        if (bestT == SKIP) {
            if (c.must) return false;
            continue;
        }
        bool include = c.must || bestE.score > cur.score || (c.restaurant && prof_.lunch);
        if (!include) continue;
        insertSorted(tl, i, bestT);
        cur = bestE;
        if (isActivity(i)) ++acts;
        if (c.restaurant) lunch = true;
    }
    bool hasRestaurant = false;
    for (int i = 0; i < n_; ++i) hasRestaurant |= cands_[i].restaurant;
    if (prof_.lunch && hasRestaurant && !lunch) return false;
    greedyFound_ = true;
    greedyScore_ = cur.score;
    if (cur.score > bestScore_) {
        bestScore_ = cur.score;
        best_ = tl;
        haveBest_ = true;
    }
    return true;
}

void CspSolver::solveOnce() {
    buildDomains();
    ac3Removed_ += ac3();
    assign_.assign(n_, -2);
    tl_.clear();
    includedActs_ = 0;
    lunchDone_ = false;
    curEval_ = evaluate(tl_);
    stop_ = false;
    for (int i = 0; i < n_; ++i)
        if (cands_[i].must && dom_[i].empty()) return;  // trivially unsolvable
    greedy();
    if (!curEval_.ok) return;
    dfs();
}

Json CspSolver::conflicts() {
    Json out = Json::array();
    double mustPrice = 0;
    for (int i = 0; i < n_; ++i) {
        if (!cands_[i].must) continue;
        mustPrice += cands_[i].price;
        if (dom_[i].empty()) {
            Json c = Json::object();
            c["type"] = "must_visit_unavailable";
            c["poi"] = cands_[i].node;
            c["name"] = city_.pois[cands_[i].poi].name;
            c["detail"] = unaryReason_[i].empty() ? std::string("no feasible time slot") : unaryReason_[i];
            out.push(c);
        }
    }
    if (mustPrice > prof_.budget) {
        Json c = Json::object();
        c["type"] = "budget";
        c["detail"] = "must-visit entry fees exceed the budget";
        out.push(c);
    }
    if (out.size() == 0) {
        // Which must-visits cannot be combined? Drop each in turn and test greedily.
        std::vector<Candidate> saved = cands_;
        for (int i = 0; i < n_; ++i) {
            if (!saved[i].must) continue;
            cands_ = saved;
            cands_[i].must = false;
            cands_[i].utility = -1e6;  // effectively removed
            bestScore_ = kNegInf;
            haveBest_ = false;
            buildDomains();
            assign_.assign(n_, -2);
            if (greedy()) {
                Json c = Json::object();
                c["type"] = "must_visit_conflict";
                c["poi"] = saved[i].node;
                c["name"] = city_.pois[saved[i].poi].name;
                c["detail"] = "cannot be combined with the other fixed activities in the available time";
                out.push(c);
            }
        }
        cands_ = saved;
        haveBest_ = false;
    }
    if (out.size() == 0) {
        Json c = Json::object();
        c["type"] = "time_window";
        c["detail"] = "no combination of activities fits the day window, budget and walking limit";
        out.push(c);
    }
    return out;
}

Json CspSolver::buildOutput(const char* status) {
    Json out = Json::object();
    out["status"] = status;
    Json acts = Json::array();
    Json legsOut = Json::array();
    Json soft = Json::array();
    double utility = 0, crowdPen = 0;
    if (haveBest_) {
        Eval e = evaluate(best_);
        int curNode = prof_.start_node;
        double curT = prof_.start_min;
        auto addLeg = [&](int from, int to, double depart, int modes) {
            const LegRouter::Leg cached = legs_.leg(from, to, depart, modes);
            RouteResult r = legs_.full(from, to, depart, modes);
            Json lj = routeToJson(r, city_, false);
            // leg-level numbers come from the same cache the solver used, so they match the schedule
            lj["depart"] = depart;
            lj["arrive"] = depart + cached.time;
            lj["time_min"] = cached.time;
            lj["money"] = cached.money;
            lj["co2_kg"] = cached.co2;
            lj["walk_m"] = cached.walk;
            lj["transfers"] = cached.transfers;
            lj["from_node"] = from;
            lj["to_node"] = to;
            legsOut.push(lj);
            return cached;
        };
        for (const Item& it : best_) {
            const Candidate& c = cands_[it.c];
            const Poi& poi = city_.pois[c.poi];
            addLeg(curNode, c.node, curT, modesFor(it.c));
            int s = slotIndex(it.start);
            double wait = vwait_[static_cast<std::size_t>(it.c) * nSlots_ + s];
            CrowdEstimate ce = estimateCrowd(city_, ctx_, c.poi, it.start, it.start + c.dur, prof_.crowd_threshold, w_.planned_load);
            Json a = Json::object();
            a["poi"] = c.node;
            a["name"] = poi.name;
            a["kind"] = poi.kind;
            a["start"] = it.start;
            a["wait"] = wait;
            a["visit_start"] = it.start + wait;
            a["end"] = it.start + wait + c.dur;
            a["utility"] = c.utility;
            a["price"] = c.price;
            a["rho"] = ce.rho_max;
            a["pvh"] = ce.pvh;
            a["local"] = c.local;
            a["must"] = c.must;
            acts.push(a);
            utility += c.utility;
            crowdPen += ce.penalty;
            if (ce.rho_max > prof_.crowd_threshold) {
                Json sv = Json::object();
                sv["type"] = "crowd_threshold";
                sv["poi"] = c.node;
                sv["name"] = poi.name;
                sv["value"] = ce.rho_max;
                sv["limit"] = prof_.crowd_threshold;
                soft.push(sv);
            }
            curNode = c.node;
            curT = it.start + wait + c.dur;
        }
        LegRouter::Leg back = addLeg(curNode, prof_.end_node, curT, prof_.modes);
        if (e.co2 > prof_.co2_target_kg) {
            Json sv = Json::object();
            sv["type"] = "co2_target";
            sv["value"] = e.co2;
            sv["limit"] = prof_.co2_target_kg;
            soft.push(sv);
        }
        if (e.transfers > prof_.max_transfers) {
            Json sv = Json::object();
            sv["type"] = "transfers";
            sv["value"] = e.transfers;
            sv["limit"] = prof_.max_transfers;
            soft.push(sv);
        }
        out["score"] = e.score;
        Json tot = Json::object();
        tot["utility"] = utility;
        tot["money"] = e.money;
        tot["co2_kg"] = e.co2;
        tot["walk_m"] = e.walk;
        tot["travel_min"] = e.travel;
        tot["wait_min"] = e.wait;
        tot["transfers"] = e.transfers;
        tot["crowd_penalty"] = crowdPen;
        tot["end_time"] = curT + back.time;
        out["totals"] = tot;
    } else {
        out["conflicts"] = conflicts();
    }
    out["activities"] = acts;
    out["legs"] = legsOut;
    out["soft_violations"] = soft;
    Json rel = Json::array();
    for (const auto& r : relaxed_) rel.push(r);
    out["relaxed"] = rel;
    Json st = Json::object();
    st["nodes"] = static_cast<double>(nodes_);
    st["backtracks"] = static_cast<double>(backtracks_);
    st["pruned"] = static_cast<double>(pruned_);
    st["fc_removed"] = static_cast<double>(fcRemoved_);
    st["ac3_removed"] = ac3Removed_;
    st["greedy_found"] = greedyFound_;
    st["greedy_score"] = greedyScore_;
    st["route_cache"] = static_cast<double>(legs_.cacheSize());
    st["route_searches"] = static_cast<double>(legs_.searches());
    st["complete_search"] = !stop_;
    st["runtime_ms"] = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0_).count();
    out["stats"] = st;
    return out;
}

Json CspSolver::run() {
    t0_ = std::chrono::steady_clock::now();
    solveOnce();
    // Soft constraints that were made hard are relaxed one at a time, in order.
    for (const auto& name : opt_.relax_order) {
        if (haveBest_) break;
        if (name == "co2_target" && opt_.soft_as_hard_co2) opt_.soft_as_hard_co2 = false;
        else if (name == "crowd_threshold" && opt_.soft_as_hard_crowd) opt_.soft_as_hard_crowd = false;
        else continue;
        relaxed_.push_back(name);
        solveOnce();
    }
    return buildOutput(haveBest_ ? "ok" : "unsolvable");
}

}  // namespace

Json solveItinerary(const City& city, const Context* ctx, const Json& req) {
    Profile prof = Profile::fromJson(req["profile"]);
    Weights w = Weights::fromJson(req["weights"]);
    CspOptions opt;
    const Json& o = req["options"];
    opt.node_limit = static_cast<long>(o.getNum("node_limit", static_cast<double>(opt.node_limit)));
    opt.time_limit_ms = o.getNum("time_limit_ms", opt.time_limit_ms);
    opt.value_limit = o.getInt("value_limit", opt.value_limit);
    opt.soft_as_hard_crowd = o.getBool("soft_as_hard_crowd", false);
    opt.soft_as_hard_co2 = o.getBool("soft_as_hard_co2", false);
    if (o["relax_order"].isArray()) {
        opt.relax_order.clear();
        for (const auto& r : o["relax_order"].items()) opt.relax_order.push_back(r.str());
    }
    std::vector<Candidate> cands;
    for (const auto& cj : req["candidates"].items()) {
        Candidate c;
        c.node = cj.getInt("poi", -1);
        auto it = city.poiIndex.find(c.node);
        if (it == city.poiIndex.end()) continue;
        c.poi = it->second;
        const Poi& poi = city.pois[c.poi];
        c.utility = cj.getNum("utility", 1.0);
        c.dur = cj.getInt("duration_min", static_cast<int>(poi.duration_min));
        c.price = cj.getNum("price", poi.price);
        c.must = cj.getBool("must", false);
        c.restaurant = cj.getBool("restaurant", poi.kind == "restaurant");
        c.local = cj.getBool("local", poi.independent);
        c.modes_override = cj.has("modes") ? modesFromJson(cj["modes"], -1) : -1;
        for (const auto& f : cj["forbid"].items()) c.forbid.emplace_back(f.at(0).integer(), f.at(1).integer());
        cands.push_back(c);
    }
    CspSolver solver(city, ctx, prof, w, cands, opt);
    return solver.run();
}

}  // namespace tp
