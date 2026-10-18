// Unit tests for the C++ engine on a tiny hand-made city.
#include <cmath>
#include <iostream>
#include <string>

#include "engine.hpp"
#include "json.hpp"

using tp::Json;

static int failures = 0, checks = 0;
#define CHECK(cond, msg)                                                  \
    do {                                                                  \
        ++checks;                                                         \
        if (!(cond)) {                                                    \
            ++failures;                                                   \
            std::cerr << "  FAIL: " << msg << "  (" #cond ")\n";          \
        }                                                                 \
    } while (0)

// Five places in a row (0 hotel, 1..3 POIs, 4 station) plus one transit line.
static const char* kCity = R"({
 "slot_min": 15, "day_start_min": 480, "day_end_min": 1260, "themes": ["a"],
 "params": {"walk_m_per_min": 75, "car_kmh_offpeak": 30, "car_kmh_peak": 15, "peak_windows": [[510, 630]]},
 "nodes": [
  {"id": 0, "name": "Hotel", "type": "hotel", "x": 0, "y": 0, "zone": 0},
  {"id": 1, "name": "Museum", "type": "poi", "x": 1000, "y": 0, "zone": 0},
  {"id": 2, "name": "Garden", "type": "poi", "x": 2000, "y": 0, "zone": 0},
  {"id": 3, "name": "Fort", "type": "poi", "x": 3000, "y": 0, "zone": 1, "accessible": false},
  {"id": 4, "name": "Station", "type": "stop", "x": 3000, "y": 300, "zone": 1}],
 "lines": [{"id": "L1", "name": "Line 1", "kind": "metro", "speed_kmh": 30, "headway_min": 6, "fare_base": 10,
            "fare_per_km": 1, "co2_g_pkm": 30}],
 "edges": [
  {"u": 0, "v": 1, "mode": "walk", "dist_m": 1100}, {"u": 1, "v": 2, "mode": "walk", "dist_m": 1100},
  {"u": 2, "v": 3, "mode": "walk", "dist_m": 1100, "accessible": false}, {"u": 3, "v": 4, "mode": "walk", "dist_m": 350},
  {"u": 0, "v": 4, "mode": "transit", "line": "L1", "dist_m": 3300},
  {"u": 0, "v": 2, "mode": "car", "dist_m": 2400}, {"u": 2, "v": 3, "mode": "car", "dist_m": 1200}],
 "pois": [
  {"node": 1, "name": "Museum", "kind": "attraction", "capacity": 50, "duration_min": 60, "price": 100, "open_min": 540, "close_min": 1080},
  {"node": 2, "name": "Garden", "kind": "attraction", "capacity": 50, "duration_min": 60, "price": 0, "open_min": 480, "close_min": 1200},
  {"node": 3, "name": "Fort", "kind": "attraction", "capacity": 50, "duration_min": 90, "price": 200, "open_min": 600, "close_min": 1020,
   "accessible": false, "ticketed": true}]
})";

static Json call(tp::Engine& e, const std::string& json) {
    return e.handle(Json::parse(json));
}

int main() {
    // JSON round trip
    Json j = Json::parse(R"({"a":[1,2.5,"x\n\"y"],"b":{"c":true,"d":null},"e":-3})");
    CHECK(j["a"].at(1).num() == 2.5, "parse number");
    CHECK(j["a"].at(2).str() == "x\n\"y", "parse escaped string");
    CHECK(Json::parse(j.dump()).dump() == j.dump(), "dump/parse round trip");
    CHECK(j["missing"].isNull(), "missing key is null");

    tp::Engine e;
    Json load = Json::object();
    load["cmd"] = "load_city";
    load["city"] = Json::parse(kCity);
    CHECK(e.handle(load).getBool("ok", false), "load city");

    // UCS, Greedy and A* agree on cost here; A* expands no more than UCS
    Json ucs = call(e, R"({"cmd":"route","from":0,"to":3,"depart":700,"algorithm":"ucs","weights":{"time":1}})");
    Json ast = call(e, R"({"cmd":"route","from":0,"to":3,"depart":700,"algorithm":"astar","weights":{"time":1}})");
    Json grd = call(e, R"({"cmd":"route","from":0,"to":3,"depart":700,"algorithm":"greedy","weights":{"time":1}})");
    CHECK(ucs.getBool("found", false) && ast.getBool("found", false) && grd.getBool("found", false), "routes found");
    CHECK(std::fabs(ucs.getNum("cost", 0) - ast.getNum("cost", 1)) < 1e-6, "A* is optimal (same cost as UCS)");
    CHECK(ast.getNum("expanded", 1e9) <= ucs.getNum("expanded", 0), "A* expands no more than UCS");
    CHECK(grd.getNum("cost", 0) >= ucs.getNum("cost", 0) - 1e-6, "greedy is never better than optimal");

    // wheelchair users cannot take the inaccessible path into the Fort
    Json acc = call(e, R"({"cmd":"route","from":0,"to":2,"depart":700,"accessible_only":true,"modes":["walk"],"weights":{"time":1}})");
    CHECK(acc.getBool("found", false), "accessible route to Garden");
    Json noacc = call(e, R"({"cmd":"route","from":0,"to":3,"depart":700,"accessible_only":true,"modes":["walk"],"weights":{"time":1}})");
    CHECK(!noacc.getBool("found", true), "no accessible walking route to the Fort");

    // the greenest objective avoids the car
    Json green = call(e, R"({"cmd":"route","from":0,"to":2,"depart":700,"weights":{"time":0.01,"co2":100}})");
    bool usesCar = false;
    for (const auto& l : green["legs"].items()) usesCar |= l.getStr("mode", "") == "car";
    CHECK(!usesCar, "CO2-weighted route avoids the car");

    // a suspended line is not used
    call(e, R"({"cmd":"set_context","key":"k","conditions":{"line_delays":[{"line":"L1","from":0,"to":1440,"suspended":true}]}})");
    Json susp = call(e, R"({"cmd":"route","from":0,"to":4,"depart":700,"context":"k","weights":{"time":1}})");
    bool usesLine = false;
    for (const auto& l : susp["legs"].items()) usesLine |= l.has("line");
    CHECK(!usesLine, "suspended transit line is avoided");

    // CSP: feasible itinerary respects opening hours and the day window
    call(e, R"({"cmd":"set_context","key":"c","reset_planned":true})");
    std::string solve = R"({"cmd":"solve","context":"c","weights":{"utility":1,"time":0.02},
      "profile":{"start_node":0,"end_node":0,"start_min":540,"end_min":1080,"budget":1000,"max_walk_m":20000,
                 "modes":["walk","transit","car"],"max_activities":3,"lunch":false},
      "candidates":[{"poi":1,"utility":5},{"poi":2,"utility":4},{"poi":3,"utility":8,"must":true}]})";
    Json sol = call(e, solve);
    CHECK(sol.getStr("status", "") == "ok", "CSP solves");
    bool hasFort = false;
    double prevEnd = 540;
    for (const auto& a : sol["activities"].items()) {
        hasFort |= a.getInt("poi", -1) == 3;
        CHECK(a.getNum("start", 0) >= prevEnd - 1e-6, "activities do not overlap");
        prevEnd = a.getNum("end", 0);
    }
    CHECK(hasFort, "must-visit is scheduled");
    CHECK(sol["totals"].getNum("end_time", 9999) <= 1080 + 1e-6, "itinerary ends inside the day window");

    // CSP: a closed must-visit makes the request unsolvable, with the reason
    call(e, R"({"cmd":"set_context","key":"c","conditions":{"closed_pois":[3]}})");
    Json bad = call(e, solve);
    CHECK(bad.getStr("status", "") == "unsolvable", "closed must-visit -> unsolvable");
    CHECK(bad["conflicts"].size() > 0 && bad["conflicts"].at(0).getStr("type", "") == "must_visit_unavailable",
          "conflict names the unavailable must-visit");

    // Planner: plans the CSP goals; replanning avoids a newly closed place
    call(e, R"({"cmd":"set_context","key":"p","reset_planned":true,"conditions":{}})");
    std::string plan = R"({"cmd":"plan","context":"p","weights":{"utility":1,"time":0.02},
      "profile":{"start_node":0,"end_node":0,"start_min":540,"end_min":1080,"budget":1000,"max_walk_m":20000,"lunch":false},
      "goals":[{"poi":1,"utility":5},{"poi":2,"utility":4}],"hint_order":[1,2],"state":{"node":0,"time":540,"budget":1000}})";
    Json p = call(e, plan);
    CHECK(p["visited"].size() == 2, "planner visits both goals");
    call(e, R"({"cmd":"set_context","key":"p","conditions":{"closed_pois":[2]}})");
    Json rp = call(e, R"({"cmd":"replan","context":"p","weights":{"utility":1,"time":0.02},
      "profile":{"start_node":0,"end_node":0,"start_min":540,"end_min":1080,"budget":1000,"max_walk_m":20000,"lunch":false},
      "goals":[{"poi":2,"utility":4}],"alternatives":[{"poi":3,"utility":3,"replaces":2}],"previous_order":[2],
      "state":{"node":1,"time":660,"budget":900}})");
    bool visitsClosed = false;
    for (const auto& v : rp["visited"].items()) visitsClosed |= v.integer() == 2;
    CHECK(!visitsClosed, "replan does not visit the closed place");
    CHECK(rp.getInt("conflicts_before", 0) > 0, "replan detected the conflict");

    std::cout << (checks - failures) << "/" << checks << " engine checks passed\n";
    return failures == 0 ? 0 : 1;
}
