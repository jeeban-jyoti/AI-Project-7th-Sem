"""Talks to the C++ engine over a JSON-lines pipe.

The engine runs as one long-lived subprocess per Python process: it loads the
city once and then answers route / matrix / solve / plan / replan requests.
"""
from __future__ import annotations

import json
import os
import subprocess
from typing import Any, Dict, Iterable, List, Optional

from tourism.core.config import engine_path, load_config


class EngineError(RuntimeError):
    pass


class EngineClient:
    def __init__(self, exe: Optional[str] = None, city_path: Optional[str] = None):
        exe = exe or engine_path(load_config())
        if not os.path.exists(exe):
            raise EngineError("C++ engine not found at %s - build it first with ./run.sh build" % exe)
        self.exe = exe
        self.proc = subprocess.Popen([exe], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                     stderr=subprocess.PIPE, universal_newlines=True, bufsize=1,
                                     encoding="utf-8")
        self._id = 0
        self.calls = 0
        if city_path:
            self.call("load_city", path=os.path.abspath(city_path))

    def call(self, cmd: str, **kwargs: Any) -> Dict[str, Any]:
        if self.proc.poll() is not None:
            raise EngineError("engine process has exited")
        self._id += 1
        req = dict(kwargs)
        req["cmd"] = cmd
        req["id"] = self._id
        assert self.proc.stdin is not None and self.proc.stdout is not None
        self.proc.stdin.write(json.dumps(req, separators=(",", ":")) + "\n")
        self.proc.stdin.flush()
        line = self.proc.stdout.readline()
        if not line:
            err = self.proc.stderr.read() if self.proc.stderr else ""
            raise EngineError("engine terminated unexpectedly: %s" % err.strip())
        resp = json.loads(line)
        self.calls += 1
        if not resp.get("ok", False):
            raise EngineError("%s failed: %s" % (cmd, resp.get("error", "unknown error")))
        return resp

    # convenience wrappers -------------------------------------------------
    def route(self, src: int, dst: int, depart: int, modes: Iterable[str], weights: Dict[str, float],
              algorithm: str = "astar", accessible_only: bool = False, context: Optional[str] = None) -> Dict[str, Any]:
        kw: Dict[str, Any] = dict(**{"from": src}, to=dst, depart=depart, modes=list(modes), weights=weights,
                                  algorithm=algorithm, accessible_only=accessible_only)
        if context:
            kw["context"] = context
        return self.call("route", **kw)

    def matrix(self, nodes: List[int], depart: int, modes: Iterable[str], weights: Dict[str, float],
               context: Optional[str] = None) -> Dict[str, Any]:
        kw: Dict[str, Any] = dict(nodes=nodes, depart=depart, modes=list(modes), weights=weights)
        if context:
            kw["context"] = context
        return self.call("matrix", **kw)

    def set_context(self, key: str, load: List[List[float]], wait: List[List[float]], pvh: List[List[float]],
                    conditions: Optional[Dict[str, Any]] = None, reset_planned: bool = False) -> None:
        self.call("set_context", key=key, load=load, wait=wait, pvh=pvh, conditions=conditions or {},
                  reset_planned=reset_planned)

    def set_conditions(self, key: str, conditions: Dict[str, Any]) -> None:
        self.call("set_context", key=key, conditions=conditions)

    def add_load(self, key: str, items: List[List[float]], clear: bool = False) -> None:
        self.call("add_load", key=key, items=items, clear=clear)

    def close(self) -> None:
        if self.proc.poll() is None:
            try:
                assert self.proc.stdin is not None
                self.proc.stdin.write('{"cmd":"quit"}\n')
                self.proc.stdin.flush()
                self.proc.wait(timeout=2)
            except Exception:
                self.proc.kill()
        for stream in (self.proc.stdin, self.proc.stdout, self.proc.stderr):
            try:
                if stream:
                    stream.close()
            except Exception:
                pass

    def __enter__(self) -> "EngineClient":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass
