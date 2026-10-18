// tourism_engine: JSON-lines server.
//
//   tourism_engine [--city data/city.json]              serve requests on stdin/stdout
//   tourism_engine --city data/city.json --request r.json   answer one request and exit
//
// Each input line is one JSON request {"id":..,"cmd":"route"|"solve"|"plan"|...};
// each output line is the JSON response with the same id.
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

#include "engine.hpp"

int main(int argc, char** argv) {
    std::ios::sync_with_stdio(false);
    tp::Engine engine;
    std::string requestFile;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--city" && i + 1 < argc) {
            tp::Json req = tp::Json::object();
            req["cmd"] = "load_city";
            req["path"] = argv[++i];
            tp::Json res = engine.handle(req);
            if (!res.getBool("ok", false)) {
                std::cerr << "failed to load city: " << res.getStr("error", "?") << "\n";
                return 1;
            }
        } else if (a == "--request" && i + 1 < argc) {
            requestFile = argv[++i];
        } else if (a == "--help" || a == "-h") {
            std::cout << "usage: tourism_engine [--city FILE] [--request FILE]\n";
            return 0;
        }
    }
    if (!requestFile.empty()) {
        std::ifstream in(requestFile);
        if (!in) { std::cerr << "cannot open " << requestFile << "\n"; return 1; }
        std::stringstream ss;
        ss << in.rdbuf();
        std::cout << engine.handle(tp::Json::parse(ss.str())).dump() << std::endl;
        return 0;
    }
    std::string line;
    while (std::getline(std::cin, line)) {
        if (line.empty()) continue;
        tp::Json res;
        try {
            tp::Json req = tp::Json::parse(line);
            if (req.getStr("cmd", "") == "quit") break;
            res = engine.handle(req);
        } catch (const std::exception& e) {
            res = tp::Json::object();
            res["ok"] = false;
            res["error"] = e.what();
        }
        std::cout << res.dump() << '\n' << std::flush;
    }
    return 0;
}
