// Module 2 (planning): itinerary generation as a state-space planning problem.
//
// State:   location, clock, budget, energy, hunger, walked distance,
//          goals done / skipped / shifted, lunch taken.
// Actions (preconditions -> effects, duration, resources):
//   travel(a,b)          route exists, walk limit, budget  -> at b, clock+leg, money, CO2, energy-
//   buy_ticket(p)        at p, ticketed, budget>=price     -> ticket, money-, 5 min
//   visit(p)             at p, open, below capacity,
//                        energy>=20, hunger<85             -> done(p), clock+wait+duration
//   eat(r)               at r, lunch window or hungry       -> hunger=0, energy+
//   local_experience(x)  like visit, independent business  -> done(x), local support
//   shift_visit(p)       at p, crowded now, quieter slot
//                        later before closing              -> clock moves to the off-peak slot
//   rest                 energy<40                          -> energy+40, 30 min
//   skip(p)              p not a must-visit                 -> p dropped (cost = lost utility)
//   finish               all goals resolved                 -> travel to end before day end
//
// plan():   A* (h = weighted straight-line time to the end node) over this space.
// replan(): keep -> min-conflicts local repair -> full A* replan.
#pragma once

#include "csp.hpp"

namespace tp {

Json planItinerary(const City& city, const Context* ctx, const Json& request);
Json replanItinerary(const City& city, const Context* ctx, const Json& request);

}  // namespace tp
