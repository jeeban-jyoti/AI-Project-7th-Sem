#!/usr/bin/env bash
# Sustainable and Crowd-Aware Tourism Planner - one script to build and run everything.
#
#   ./run.sh all            setup -> build -> data -> test -> train -> bench -> demo -> evaluate -> report
#   ./run.sh help           list every command
#
# Works with macOS's default bash 3.2, Linux and WSL / Git Bash. Needs Python 3.8+
# and a C++17 compiler (clang++ or g++); no Python packages are required.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONDONTWRITEBYTECODE=1
BUILD_DIR="$ROOT/build"
ENGINE="$BUILD_DIR/tourism_engine"
ENGINE_TESTS="$BUILD_DIR/engine_tests"

bold() { printf '\n\033[1m%s\033[0m\n' "$*"; }
info() { printf '  %s\n' "$*"; }
die()  { printf '\nERROR: %s\n' "$*" >&2; exit 1; }

find_python() {
  if [ -n "${PYTHON:-}" ]; then echo "$PYTHON"; return; fi
  for c in python3 python; do
    if command -v "$c" >/dev/null 2>&1 && "$c" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 8) else 1)' 2>/dev/null; then
      echo "$c"; return
    fi
  done
  echo ""
}

find_cxx() {
  if [ -n "${CXX:-}" ]; then echo "$CXX"; return; fi
  for c in c++ clang++ g++; do
    if command -v "$c" >/dev/null 2>&1; then echo "$c"; return; fi
  done
  echo ""
}

PY="$(find_python)"
CXXBIN="$(find_cxx)"

cmd_setup() {
  bold "Checking the toolchain"
  [ -n "$PY" ] || die "Python 3.8+ not found. Install it (macOS: 'xcode-select --install' or python.org; Ubuntu: 'sudo apt install python3')."
  info "python:   $("$PY" --version 2>&1) ($(command -v "$PY"))"
  [ -n "$CXXBIN" ] || die "No C++ compiler found. macOS: 'xcode-select --install'; Ubuntu: 'sudo apt install g++'; Windows: use WSL."
  info "compiler: $("$CXXBIN" --version 2>&1 | head -n 1)"
  info "No Python packages needed (standard library only)."
}

cmd_build() {
  [ -n "$CXXBIN" ] || die "No C++ compiler found (see ./run.sh setup)."
  bold "Building the C++ engine (search, CSP, planner)"
  mkdir -p "$BUILD_DIR"
  local SRC_ARR=()
  local f
  for f in "$ROOT"/cpp/src/*.cpp; do
    [ "$(basename "$f")" = "main.cpp" ] && continue
    SRC_ARR+=("$f")
  done
  "$CXXBIN" -std=c++17 -O2 -Wall -Wextra -I"$ROOT/cpp/include" "${SRC_ARR[@]}" "$ROOT/cpp/src/main.cpp" -o "$ENGINE"
  info "built build/tourism_engine"
  "$CXXBIN" -std=c++17 -O2 -Wall -Wextra -I"$ROOT/cpp/include" "${SRC_ARR[@]}" "$ROOT/cpp/tests/test_engine.cpp" -o "$ENGINE_TESTS"
  info "built build/engine_tests"
}

# rebuild when the binary is missing, was built on another platform, or the sources changed
need_engine() {
  if [ ! -x "$ENGINE" ] || [ ! -x "$ENGINE_TESTS" ] || ! "$ENGINE" --help >/dev/null 2>&1 \
     || [ -n "$(find "$ROOT/cpp" -newer "$ENGINE" -type f 2>/dev/null | head -n 1)" ]; then
    cmd_build
  fi
}
need_data()   { [ -f "$ROOT/data/city.json" ] && [ -f "$ROOT/data/history.csv.gz" ] || cmd_data; }
need_models() { [ -f "$ROOT/outputs/models/forecaster.json" ] && [ -f "$ROOT/outputs/models/rl_q_learning.json" ] || cmd_train; }

cmd_data()  { [ -n "$PY" ] || die "Python not found"; "$PY" -m tourism.cli data; }
cmd_train() { need_engine; need_data; "$PY" -m tourism.cli train "$@"; }

cmd_test() {
  need_engine; need_data
  bold "C++ engine tests"
  "$ENGINE_TESTS"
  bold "Python tests"
  "$PY" -m unittest discover -s "$ROOT/tests" -t "$ROOT" "$@"
}

cmd_bench()    { need_engine; need_data; [ -f "$ROOT/outputs/models/matrix.json" ] || need_models; "$PY" -m tourism.cli bench "$@"; }
cmd_demo()     { need_engine; need_data; need_models; "$PY" -m tourism.cli demo "$@"; }
cmd_plan()     { need_engine; need_data; need_models; "$PY" -m tourism.cli plan "$@"; }
cmd_simulate() { need_engine; need_data; need_models; "$PY" -m tourism.cli simulate "$@"; }
cmd_evaluate() { need_engine; need_data; need_models; "$PY" -m tourism.cli evaluate "$@"; }
cmd_tools()    { need_engine; need_data; need_models; "$PY" -m tourism.cli tools "$@"; }
cmd_pddl()     { need_engine; need_data; need_models; "$PY" -m tourism.cli pddl "$@"; }
cmd_info()     { need_data; "$PY" -m tourism.cli info "$@"; }

cmd_report() {
  local mode="quick"
  if [ "${1:-}" = "--mode" ] && [ -n "${2:-}" ]; then mode="$2"; fi
  "$PY" -m tourism.cli report --mode "$mode"
  info "Open it with:  open \"$ROOT/outputs/report/report.html\"   (Linux: xdg-open)"
}

cmd_all() {
  local start
  start=$(date +%s)
  cmd_setup
  cmd_build
  cmd_data
  cmd_test
  cmd_train
  cmd_bench
  cmd_demo
  cmd_evaluate --mode "${MODE:-quick}"
  cmd_report --mode "${MODE:-quick}"
  bold "All done in $(( $(date +%s) - start ))s"
  info "Report:   outputs/report/report.html"
  info "Results:  outputs/results/   Models: outputs/models/   Plans: outputs/plans/"
}

cmd_clean() {
  bold "Removing build/, outputs/ and generated data"
  rm -rf "$BUILD_DIR" "$ROOT/outputs" "$ROOT/data/city.json" "$ROOT/data/history.csv.gz" "$ROOT/data/transit_history.csv"
  find "$ROOT" -name "__pycache__" -type d -prune -exec rm -rf {} + 2>/dev/null || true
}

cmd_help() {
  cat <<'EOF'
Usage: ./run.sh <command> [options]

  all                     everything below in order (MODE=full ./run.sh all for the full grid)
  setup                   check Python and the C++ compiler
  build                   compile the C++ engine and its tests into build/
  data                    generate the synthetic city and 120 days of crowd history (data/)
  test                    run C++ and Python unit tests
  train [--episodes N]    learn the Bayesian networks (M3) and RL policies (M4) -> outputs/models/
  bench                   UCS vs Greedy vs A*, and CSP statistics (M1)
  demo                    conventional vs sustainable plan, an impossible request, an event day with replanning
  plan [options]          plan one traveller:
                            --profile examples/profiles/family_wheelchair.json
                            --interactive              answer a short form and accept/reject suggestions
                            --scenario S1_peak_weekend --system sustainable|conventional|min_distance|env_only
  simulate [options]      simulate one day: --scenario S3_public_event --system sustainable --travellers 120
  evaluate [--mode quick|full] [--workers N]    run all scenarios x systems -> outputs/results/
  report [--mode quick|full]                    build outputs/report/report.html
  tools                   list the GenAI tool registry (Module 5 hooks) and call a few tools
  pddl                    print the planning domain and export it as PDDL
  info                    describe the city
  clean                   delete build/, outputs/ and generated data

Scenarios: S0_normal S1_peak_weekend S2_sudden_crowd S3_public_event S4_transport_delay S5_capacity_limit S6_rain
Environment: PYTHON=/path/to/python3  CXX=clang++  TOURISM_CONFIG=config/other.json
EOF
}

cmd="${1:-help}"
[ $# -gt 0 ] && shift
case "$cmd" in
  all) cmd_all "$@" ;;
  setup) cmd_setup ;;
  build) cmd_build ;;
  data) cmd_data ;;
  test) cmd_test "$@" ;;
  train) cmd_train "$@" ;;
  bench) cmd_bench "$@" ;;
  demo) cmd_demo "$@" ;;
  plan) cmd_plan "$@" ;;
  simulate) cmd_simulate "$@" ;;
  evaluate) cmd_evaluate "$@" ;;
  report) cmd_report "$@" ;;
  tools) cmd_tools "$@" ;;
  pddl) cmd_pddl "$@" ;;
  info) cmd_info "$@" ;;
  clean) cmd_clean ;;
  help|-h|--help) cmd_help ;;
  *) echo "Unknown command: $cmd"; cmd_help; exit 1 ;;
esac
