// Engine facade: a JSON request/response API over the city, search, CSP and
// planner. main.cpp exposes it as a JSON-lines server on stdin/stdout so the
// Python side (and later the GenAI agents' tools) can call it.
#pragma once

#include <map>
#include <string>

#include "city.hpp"
#include "json.hpp"

namespace tp {

class Engine {
public:
    Json handle(const Json& request);
    bool loaded() const { return loaded_; }
    const City& city() const { return city_; }

private:
    Json cmdLoadCity(const Json& req);
    Json cmdSetContext(const Json& req);
    Json cmdAddLoad(const Json& req);
    Json cmdRoute(const Json& req);
    Json cmdMatrix(const Json& req);
    Json cmdBench(const Json& req);
    Json cmdCityInfo() const;
    const Context* contextFor(const Json& req, Conditions* inlineCond) const;

    City city_;
    bool loaded_ = false;
    std::map<std::string, Context> contexts_;
};

}  // namespace tp
