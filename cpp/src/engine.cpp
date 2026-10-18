#include "engine.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "csp.hpp"
#include "planner.hpp"
#include "search.hpp"

namespace tp {

namespace {

Json ok() {
    Json j = Json::object();
    j["ok"] = true;
    return j;
}

std::vector<std::vector<double>> readTable(const Json& arr, int rows, int cols, double def) {
    std::vector<std::vector<double>> t(rows, std::vector<double>(cols, def));
    if (!arr.isArray()) return t;
    for (int r = 0; r < rows && r < static_cast<int>(arr.size()); ++r) {
        const Json& row = arr.at(r);
        for (int c = 0; c < cols && c < static_cast<int>(row.size()); ++c) t[r][c] = row.at(c).num(def);
    }
    return t;
}

}  // namespace

const Context* Engine::contextFor(const Json& req, Conditions* inlineCond) const {
    std::string key = req.getStr("context", "");
    if (!key.empty()) {
        auto it = contexts_.find(key);
        if (it == contexts_.end()) throw std::runtime_error("unknown context: " + key);
        return &it->second;
    }
    if (inlineCond && req.has("conditions")) *inlineCond = conditionsFromJson(req["conditions"], city_);
    return nullptr;
}

Json Engine::cmdLoadCity(const Json& req) {
    Json cityJson;
    if (req.has("city")) {
        cityJson = req["city"];
    } else {
        std::string path = req.getStr("path", "");
        std::ifstream in(path);
        if (!in) throw std::runtime_error("cannot open city file: " + path);
        std::stringstream ss;
        ss << in.rdbuf();
        cityJson = Json::parse(ss.str());
    }
    city_.load(cityJson);
    contexts_.clear();
    loaded_ = true;
    Json out = ok();
    out["nodes"] = static_cast<int>(city_.nodes.size());
    out["pois"] = static_cast<int>(city_.pois.size());
    out["lines"] = static_cast<int>(city_.lines.size());
    out["slots"] = city_.nSlots();
    return out;
}

Json Engine::cmdSetContext(const Json& req) {
    std::string key = req.getStr("key", "default");
    Context& ctx = contexts_[key];
    int P = static_cast<int>(city_.pois.size());
    int S = city_.nSlots();
    ctx.n_slots = S;
    if (req.has("load")) ctx.load = readTable(req["load"], P, S, 0.0);
    if (req.has("wait")) ctx.wait = readTable(req["wait"], P, S, 0.0);
    if (req.has("pvh")) ctx.pvh = readTable(req["pvh"], P, S, 0.0);
    if (ctx.load.empty()) ctx.load.assign(P, std::vector<double>(S, 0.0));
    if (ctx.wait.empty()) ctx.wait.assign(P, std::vector<double>(S, 0.0));
    if (ctx.pvh.empty()) ctx.pvh.assign(P, std::vector<double>(S, 0.0));
    if (req.getBool("reset_planned", false) || ctx.planned.empty()) ctx.planned.assign(P, std::vector<double>(S, 0.0));
    if (req.has("conditions")) ctx.cond = conditionsFromJson(req["conditions"], city_);
    return ok();
}

Json Engine::cmdAddLoad(const Json& req) {
    auto it = contexts_.find(req.getStr("key", "default"));
    if (it == contexts_.end()) throw std::runtime_error("unknown context");
    Context& ctx = it->second;
    if (req.getBool("clear", false))
        for (auto& row : ctx.planned) std::fill(row.begin(), row.end(), 0.0);
    int added = 0;
    for (const auto& item : req["items"].items()) {
        auto pit = city_.poiIndex.find(item.at(0).integer());
        if (pit == city_.poiIndex.end()) continue;
        int t0 = item.at(1).integer(), t1 = item.at(2).integer();
        double amount = item.at(3).num(1.0);
        int s0 = city_.slotOf(t0), s1 = city_.slotOf(std::max(t0, t1 - 1));
        for (int s = s0; s <= s1; ++s) ctx.planned[pit->second][s] += amount;
        ++added;
    }
    Json out = ok();
    out["added"] = added;
    return out;
}

Json Engine::cmdRoute(const Json& req) {
    Conditions inl;
    const Context* ctx = contextFor(req, &inl);
    RouteRequest r;
    r.from = req.getInt("from", 0);
    r.to = req.getInt("to", 0);
    r.depart = req.getInt("depart", 540);
    r.modes = modesFromJson(req["modes"], kAllModes);
    r.accessible_only = req.getBool("accessible_only", false);
    r.algorithm = algorithmFromString(req.getStr("algorithm", "astar"));
    r.w = Weights::fromJson(req["weights"]);
    r.cond = ctx ? &ctx->cond : &inl;
    int n = static_cast<int>(city_.nodes.size());
    if (r.from < 0 || r.from >= n || r.to < 0 || r.to >= n) throw std::runtime_error("node out of range");
    Router router(city_);
    Json out = routeToJson(router.route(r), city_, true);
    out["ok"] = true;
    out["algorithm"] = req.getStr("algorithm", "astar");
    return out;
}

Json Engine::cmdMatrix(const Json& req) {
    Conditions inl;
    const Context* ctx = contextFor(req, &inl);
    std::vector<int> nodes;
    for (const auto& n : req["nodes"].items()) nodes.push_back(n.integer());
    RouteRequest base;
    base.depart = req.getInt("depart", 600);
    base.modes = modesFromJson(req["modes"], kAllModes);
    base.accessible_only = req.getBool("accessible_only", false);
    base.w = Weights::fromJson(req["weights"]);
    base.cond = ctx ? &ctx->cond : &inl;
    Router router(city_);
    Json time = Json::array(), cost = Json::array(), money = Json::array(), co2 = Json::array(), walk = Json::array(),
         dist = Json::array(), car = Json::array();
    for (int src : nodes) {
        base.from = src;
        base.to = src;
        std::vector<RouteResult> res = router.routeToMany(base, nodes);
        Json rt = Json::array(), rc = Json::array(), rm = Json::array(), rco = Json::array(), rw = Json::array(),
             rd = Json::array(), rcar = Json::array();
        for (const auto& r : res) {
            bool usesCar = false;
            for (const auto& l : r.legs) usesCar |= (l.mode == CAR);
            rt.push(r.found ? r.time_min : -1.0);
            rc.push(r.found ? r.cost : -1.0);
            rm.push(r.money);
            rco.push(r.co2_kg);
            rw.push(r.walk_m);
            rd.push(r.dist_m);
            rcar.push(usesCar);
        }
        time.push(rt); cost.push(rc); money.push(rm); co2.push(rco); walk.push(rw); dist.push(rd); car.push(rcar);
    }
    Json out = ok();
    out["time"] = time;
    out["cost"] = cost;
    out["money"] = money;
    out["co2_kg"] = co2;
    out["walk_m"] = walk;
    out["dist_m"] = dist;
    out["car"] = car;
    return out;
}

Json Engine::cmdBench(const Json& req) {
    int pairs = req.getInt("pairs", 200);
    unsigned seed = static_cast<unsigned>(req.getInt("seed", 7));
    std::mt19937 rng(seed);
    std::vector<int> pts;
    for (const auto& p : city_.pois) pts.push_back(p.node);
    for (std::size_t i = 0; i < city_.nodes.size(); ++i) if (city_.nodes[i].type == "hotel") pts.push_back(static_cast<int>(i));
    std::uniform_int_distribution<std::size_t> pick(0, pts.size() - 1);
    std::uniform_int_distribution<int> dep(city_.day_start, city_.day_end - 120);
    RouteRequest base;
    base.modes = modesFromJson(req["modes"], kAllModes);
    base.w = Weights::fromJson(req["weights"]);
    Router router(city_);
    const char* names[3] = {"ucs", "greedy", "astar"};
    Algorithm algs[3] = {Algorithm::UCS, Algorithm::GREEDY, Algorithm::ASTAR};
    double exp[3] = {0, 0, 0}, gen[3] = {0, 0, 0}, us[3] = {0, 0, 0}, cost[3] = {0, 0, 0}, ratio[3] = {0, 0, 0};
    int optimal[3] = {0, 0, 0}, found = 0;
    for (int k = 0; k < pairs; ++k) {
        base.from = pts[pick(rng)];
        base.to = pts[pick(rng)];
        if (base.from == base.to) { --k; continue; }
        base.depart = dep(rng);
        RouteResult res[3];
        for (int a = 0; a < 3; ++a) {
            base.algorithm = algs[a];
            res[a] = router.route(base);
        }
        if (!res[0].found) continue;
        ++found;
        for (int a = 0; a < 3; ++a) {
            exp[a] += static_cast<double>(res[a].expanded);
            gen[a] += static_cast<double>(res[a].generated);
            us[a] += res[a].runtime_us;
            cost[a] += res[a].cost;
            ratio[a] += res[0].cost > 0 ? res[a].cost / res[0].cost : 1.0;
            if (res[a].cost <= res[0].cost + 1e-6) ++optimal[a];
        }
    }
    Json out = ok();
    out["pairs"] = found;
    Json algsOut = Json::object();
    for (int a = 0; a < 3; ++a) {
        Json j = Json::object();
        double n = std::max(1, found);
        j["avg_expanded"] = exp[a] / n;
        j["avg_generated"] = gen[a] / n;
        j["avg_runtime_us"] = us[a] / n;
        j["avg_cost"] = cost[a] / n;
        j["avg_cost_ratio_vs_ucs"] = ratio[a] / n;
        j["optimal_share"] = optimal[a] / n;
        algsOut[names[a]] = j;
    }
    out["algorithms"] = algsOut;
    return out;
}

Json Engine::cmdCityInfo() const {
    Json out = ok();
    out["nodes"] = static_cast<int>(city_.nodes.size());
    out["pois"] = static_cast<int>(city_.pois.size());
    out["lines"] = static_cast<int>(city_.lines.size());
    std::size_t edges = 0;
    for (const auto& a : city_.adj) edges += a.size();
    out["directed_edges"] = static_cast<double>(edges);
    out["slots"] = city_.nSlots();
    return out;
}

Json Engine::handle(const Json& req) {
    std::string cmd = req.getStr("cmd", "");
    Json out;
    try {
        if (cmd == "ping") {
            out = ok();
            out["engine"] = "tourism_engine";
            out["version"] = "1.0";
        } else if (cmd == "load_city") {
            out = cmdLoadCity(req);
        } else {
            if (!loaded_) throw std::runtime_error("no city loaded (send load_city first)");
            if (cmd == "city_info") out = cmdCityInfo();
            else if (cmd == "set_context") out = cmdSetContext(req);
            else if (cmd == "add_load") out = cmdAddLoad(req);
            else if (cmd == "drop_context") { contexts_.erase(req.getStr("key", "")); out = ok(); }
            else if (cmd == "route") out = cmdRoute(req);
            else if (cmd == "matrix") out = cmdMatrix(req);
            else if (cmd == "bench") out = cmdBench(req);
            else if (cmd == "solve") { out = solveItinerary(city_, contextFor(req, nullptr), req); out["ok"] = true; }
            else if (cmd == "plan") { out = planItinerary(city_, contextFor(req, nullptr), req); out["ok"] = true; }
            else if (cmd == "replan") { out = replanItinerary(city_, contextFor(req, nullptr), req); out["ok"] = true; }
            else throw std::runtime_error("unknown command: " + cmd);
        }
    } catch (const std::exception& e) {
        out = Json::object();
        out["ok"] = false;
        out["error"] = e.what();
    }
    if (req.has("id")) out["id"] = req["id"];
    return out;
}

}  // namespace tp
