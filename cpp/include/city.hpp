// City model shared by search, CSP and planner: nodes, multimodal edges,
// transit lines, POIs, plus per-day conditions (events) and crowd forecasts.
#pragma once

#include <map>
#include <set>
#include <string>
#include <vector>

#include "json.hpp"

namespace tp {

enum Mode { WALK = 0, TRANSIT = 1, CAR = 2 };
constexpr int kAllModes = (1 << WALK) | (1 << TRANSIT) | (1 << CAR);

struct Node {
    int id = 0;
    std::string name;
    std::string type;  // poi | stop | hotel | junction
    double x = 0, y = 0;  // metres
    int zone = 0;
    bool accessible = true;
};

struct Edge {
    int to = 0;
    int mode = WALK;
    double dist_m = 0;
    int line = -1;  // transit line index
    bool accessible = true;
};

struct Line {
    std::string id, name, kind;  // kind: metro | bus | tram
    double speed_kmh = 30, headway_min = 8, fare_base = 10, fare_per_km = 1.5, co2_g_pkm = 30;
};

struct Poi {
    int node = 0;
    std::string name, kind;  // attraction | restaurant | shop | experience
    double popularity = 0.5, rating = 4.0, capacity = 40, duration_min = 60, price = 0;
    int open_min = 540, close_min = 1080;
    bool indoor = false, accessible = true, independent = false, ticketed = false;
    int zone = 0;
};

struct CityParams {
    double walk_mpm = 80, wheel_mpm = 60;
    double car_kmh_off = 30, car_kmh_peak = 16, car_access_min = 5;
    double taxi_base = 40, taxi_per_km = 14, car_co2_g_pkm = 150;
    std::vector<std::pair<int, int>> peaks{{510, 630}, {1020, 1170}};
};

struct LineDelay {
    int line = -1;
    double minutes = 0;
    int from = 0, to = 1440;
    bool suspended = false;
};

// Live conditions for a day: events change the network and POIs.
struct Conditions {
    std::set<int> closed_pois;              // node ids
    std::map<int, double> capacity_factor;  // node id -> factor (0..1]
    std::vector<LineDelay> delays;
    std::set<int> car_closed_zones;
    double car_congestion = 1.0;

    double lineDelay(int line, int t, bool* suspended) const;
    double capFactor(int node) const;
};

struct Weights {
    double utility = 1.0, time = 0.02, money = 0.002, crowd = 0.0, wait = 0.0, co2 = 0.0;
    double transfer = 0.1, local = 0.0, planned_load = 0.0, dist = 0.0, transit_crowd = 0.0;
    static Weights fromJson(const Json& j);
};

// Crowd forecast tables for one day (from the Python uncertainty module),
// plus the load of travellers already routed by the planner (collective view).
struct Context {
    int n_slots = 0;
    std::vector<std::vector<double>> load;     // [poi][slot] background load ratio
    std::vector<std::vector<double>> wait;     // [poi][slot] expected wait (min)
    std::vector<std::vector<double>> pvh;      // [poi][slot] P(very high crowd)
    std::vector<std::vector<double>> planned;  // [poi][slot] routed visitors already planned
    Conditions cond;
};

class City {
public:
    void load(const Json& j);

    std::vector<Node> nodes;
    std::vector<std::vector<Edge>> adj;
    std::vector<Line> lines;
    std::vector<Poi> pois;
    std::map<int, int> poiIndex;  // node id -> poi index
    std::map<std::string, int> lineIndex;
    CityParams params;
    int slot_min = 15, day_start = 480, day_end = 1260;
    double max_transit_mpm = 500;

    int nSlots() const { return (day_end - day_start) / slot_min; }
    int slotOf(int t) const;
    double euclid(int a, int b) const;
    bool isPeak(int t) const;
    double carSpeedMpm(int t, double congestion) const;
    const Poi* poiAt(int node) const;
};

Conditions conditionsFromJson(const Json& j, const City& city);

// Effective crowd ratio / wait at a POI for a visit window [t0, t1).
struct CrowdEstimate {
    double rho_max = 0;     // max effective load ratio over the window
    double rho_mean = 0;
    double wait = 0;        // expected queue wait at arrival (min)
    double penalty = 0;     // sum over slots of max(0, rho - threshold)
    double pvh = 0;
};

CrowdEstimate estimateCrowd(const City& city, const Context* ctx, int poi_idx, int t0, int t1, double threshold,
                            double planned_weight);

}  // namespace tp
