#include "city.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace tp {

double Conditions::lineDelay(int line, int t, bool* suspended) const {
    double extra = 0;
    if (suspended) *suspended = false;
    for (const auto& d : delays) {
        if (d.line != line || t < d.from || t >= d.to) continue;
        if (d.suspended && suspended) *suspended = true;
        extra += d.minutes;
    }
    return extra;
}

double Conditions::capFactor(int node) const {
    auto it = capacity_factor.find(node);
    return it == capacity_factor.end() ? 1.0 : std::max(0.05, it->second);
}

Weights Weights::fromJson(const Json& j) {
    Weights w;
    w.utility = j.getNum("utility", w.utility);
    w.time = j.getNum("time", w.time);
    w.money = j.getNum("money", w.money);
    w.crowd = j.getNum("crowd", w.crowd);
    w.wait = j.getNum("wait", w.wait);
    w.co2 = j.getNum("co2", w.co2);
    w.transfer = j.getNum("transfer", w.transfer);
    w.local = j.getNum("local", w.local);
    w.planned_load = j.getNum("planned_load", w.planned_load);
    w.dist = j.getNum("dist", w.dist);
    w.transit_crowd = j.getNum("transit_crowd", w.transit_crowd);
    return w;
}

static int modeFromString(const std::string& s) {
    if (s == "walk") return WALK;
    if (s == "transit") return TRANSIT;
    if (s == "car") return CAR;
    throw std::runtime_error("unknown mode: " + s);
}

void City::load(const Json& j) {
    nodes.clear(); adj.clear(); lines.clear(); pois.clear(); poiIndex.clear(); lineIndex.clear();
    slot_min = j.getInt("slot_min", 15);
    day_start = j.getInt("day_start_min", 480);
    day_end = j.getInt("day_end_min", 1260);

    const Json& p = j["params"];
    params.walk_mpm = p.getNum("walk_m_per_min", params.walk_mpm);
    params.wheel_mpm = p.getNum("wheelchair_m_per_min", params.wheel_mpm);
    params.car_kmh_off = p.getNum("car_kmh_offpeak", params.car_kmh_off);
    params.car_kmh_peak = p.getNum("car_kmh_peak", params.car_kmh_peak);
    params.car_access_min = p.getNum("car_access_min", params.car_access_min);
    params.taxi_base = p.getNum("taxi_base", params.taxi_base);
    params.taxi_per_km = p.getNum("taxi_per_km", params.taxi_per_km);
    params.car_co2_g_pkm = p.getNum("car_co2_g_pkm", params.car_co2_g_pkm);
    if (p["peak_windows"].isArray()) {
        params.peaks.clear();
        for (const auto& w : p["peak_windows"].items()) params.peaks.emplace_back(w.at(0).integer(), w.at(1).integer());
    }

    for (const auto& n : j["nodes"].items()) {
        Node node;
        node.id = n.getInt("id", static_cast<int>(nodes.size()));
        if (node.id != static_cast<int>(nodes.size())) throw std::runtime_error("node ids must be 0..N-1 in order");
        node.name = n.getStr("name", "");
        node.type = n.getStr("type", "junction");
        node.x = n.getNum("x", 0);
        node.y = n.getNum("y", 0);
        node.zone = n.getInt("zone", 0);
        node.accessible = n.getBool("accessible", true);
        nodes.push_back(node);
    }
    adj.assign(nodes.size(), {});

    max_transit_mpm = 0;
    for (const auto& l : j["lines"].items()) {
        Line line;
        line.id = l.getStr("id", "");
        line.name = l.getStr("name", line.id);
        line.kind = l.getStr("kind", "metro");
        line.speed_kmh = l.getNum("speed_kmh", line.speed_kmh);
        line.headway_min = l.getNum("headway_min", line.headway_min);
        line.fare_base = l.getNum("fare_base", line.fare_base);
        line.fare_per_km = l.getNum("fare_per_km", line.fare_per_km);
        line.co2_g_pkm = l.getNum("co2_g_pkm", line.co2_g_pkm);
        lineIndex[line.id] = static_cast<int>(lines.size());
        max_transit_mpm = std::max(max_transit_mpm, line.speed_kmh * 1000.0 / 60.0);
        lines.push_back(line);
    }

    for (const auto& e : j["edges"].items()) {
        int u = e.getInt("u", -1), v = e.getInt("v", -1);
        if (u < 0 || v < 0 || u >= static_cast<int>(nodes.size()) || v >= static_cast<int>(nodes.size()))
            throw std::runtime_error("edge endpoint out of range");
        Edge ed;
        ed.mode = modeFromString(e.getStr("mode", "walk"));
        ed.dist_m = std::max(e.getNum("dist_m", 0), euclid(u, v));  // keeps A* admissible
        ed.accessible = e.getBool("accessible", true);
        if (ed.mode == TRANSIT) {
            auto it = lineIndex.find(e.getStr("line", ""));
            if (it == lineIndex.end()) throw std::runtime_error("transit edge with unknown line");
            ed.line = it->second;
        }
        ed.to = v;
        adj[u].push_back(ed);
        ed.to = u;
        adj[v].push_back(ed);
    }

    for (const auto& q : j["pois"].items()) {
        Poi poi;
        poi.node = q.getInt("node", -1);
        poi.name = q.getStr("name", "");
        poi.kind = q.getStr("kind", "attraction");
        poi.popularity = q.getNum("popularity", 0.5);
        poi.rating = q.getNum("rating", 4.0);
        poi.capacity = std::max(1.0, q.getNum("capacity", 40));
        poi.duration_min = q.getNum("duration_min", 60);
        poi.price = q.getNum("price", 0);
        poi.open_min = q.getInt("open_min", 540);
        poi.close_min = q.getInt("close_min", 1080);
        poi.indoor = q.getBool("indoor", false);
        poi.accessible = q.getBool("accessible", true);
        poi.independent = q.getBool("independent", false);
        poi.ticketed = q.getBool("ticketed", false);
        poi.zone = q.getInt("zone", 0);
        poiIndex[poi.node] = static_cast<int>(pois.size());
        pois.push_back(poi);
    }
}

int City::slotOf(int t) const {
    int s = (t - day_start) / slot_min;
    return std::max(0, std::min(nSlots() - 1, s));
}

double City::euclid(int a, int b) const {
    double dx = nodes[a].x - nodes[b].x, dy = nodes[a].y - nodes[b].y;
    return std::sqrt(dx * dx + dy * dy);
}

bool City::isPeak(int t) const {
    for (const auto& w : params.peaks) if (t >= w.first && t < w.second) return true;
    return false;
}

double City::carSpeedMpm(int t, double congestion) const {
    double kmh = isPeak(t) ? params.car_kmh_peak : params.car_kmh_off;
    return kmh * 1000.0 / 60.0 / std::max(1.0, congestion);
}

const Poi* City::poiAt(int node) const {
    auto it = poiIndex.find(node);
    return it == poiIndex.end() ? nullptr : &pois[it->second];
}

Conditions conditionsFromJson(const Json& j, const City& city) {
    Conditions c;
    for (const auto& v : j["closed_pois"].items()) c.closed_pois.insert(v.integer());
    for (const auto& kv : j["capacity_factor"].fields()) c.capacity_factor[std::stoi(kv.first)] = kv.second.num(1.0);
    for (const auto& d : j["line_delays"].items()) {
        LineDelay ld;
        auto it = city.lineIndex.find(d.getStr("line", ""));
        if (it == city.lineIndex.end()) continue;
        ld.line = it->second;
        ld.minutes = d.getNum("minutes", 0);
        ld.from = d.getInt("from", 0);
        ld.to = d.getInt("to", 1440);
        ld.suspended = d.getBool("suspended", false);
        c.delays.push_back(ld);
    }
    for (const auto& z : j["car_closed_zones"].items()) c.car_closed_zones.insert(z.integer());
    c.car_congestion = j.getNum("car_congestion", 1.0);
    return c;
}

CrowdEstimate estimateCrowd(const City& city, const Context* ctx, int poi_idx, int t0, int t1, double threshold,
                            double planned_weight) {
    CrowdEstimate est;
    if (!ctx || ctx->load.empty()) return est;
    const Poi& poi = city.pois[poi_idx];
    double capf = ctx->cond.capFactor(poi.node);
    int s0 = city.slotOf(t0), s1 = city.slotOf(std::max(t0, t1 - 1));
    int n = 0;
    for (int s = s0; s <= s1; ++s) {
        double rho = ctx->load[poi_idx][s] / capf;
        double base_rho = rho;
        if (planned_weight > 0 && !ctx->planned.empty())
            rho += planned_weight * ctx->planned[poi_idx][s] / (poi.capacity * capf);
        est.rho_max = std::max(est.rho_max, rho);
        est.rho_mean += rho;
        est.penalty += std::max(0.0, rho - threshold);
        if (s == s0) {
            // queue wait: forecast wait plus extra congestion caused by planned travellers
            // forecast waits already account for reduced capacity (the backlog accumulates);
            // add the extra queue caused by travellers this planner has routed here
            est.wait = ctx->wait[poi_idx][s] + city.slot_min * 2.0 * std::max(0.0, rho - std::max(base_rho, 0.8));
            est.pvh = ctx->pvh.empty() ? 0.0 : ctx->pvh[poi_idx][s];
        }
        ++n;
    }
    if (n) est.rho_mean /= n;
    return est;
}

}  // namespace tp
