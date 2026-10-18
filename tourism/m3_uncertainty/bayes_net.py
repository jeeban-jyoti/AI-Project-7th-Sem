"""A small, dependency-free discrete Bayesian network.

- CPTs are learned by counting with Dirichlet smoothing. A node can declare a
  `backoff` parent subset: its CPT rows are then smoothed towards the coarser
  distribution P(child | backoff parents) (a hierarchical Dirichlet prior), so
  rare parent combinations (holiday + event + rain) still get sensible rows.
- Inference is exact variable elimination with barren-node pruning, and
  supports soft (virtual) evidence such as a 40% chance of rain.
"""
from __future__ import annotations

import itertools
import json
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


class Factor:
    def __init__(self, variables: Sequence[str], cards: Sequence[int], values: Sequence[float]):
        self.vars = list(variables)
        self.cards = list(cards)
        self.values = list(values)
        self.strides = []
        s = 1
        for c in reversed(self.cards):
            self.strides.insert(0, s)
            s *= c
        assert len(self.values) == s, "factor size mismatch"

    def _assignments(self) -> Iterable[Tuple[int, ...]]:
        return itertools.product(*[range(c) for c in self.cards])

    def product(self, other: "Factor") -> "Factor":
        vars_ = list(self.vars)
        cards = list(self.cards)
        for v, c in zip(other.vars, other.cards):
            if v not in vars_:
                vars_.append(v)
                cards.append(c)
        ia = [vars_.index(v) for v in self.vars]
        ib = [vars_.index(v) for v in other.vars]
        values = []
        for asg in itertools.product(*[range(c) for c in cards]):
            a = sum(asg[i] * st for i, st in zip(ia, self.strides))
            b = sum(asg[i] * st for i, st in zip(ib, other.strides))
            values.append(self.values[a] * other.values[b])
        return Factor(vars_, cards, values)

    def marginalize(self, var: str) -> "Factor":
        k = self.vars.index(var)
        vars_ = self.vars[:k] + self.vars[k + 1:]
        cards = self.cards[:k] + self.cards[k + 1:]
        out = Factor(vars_, cards, [0.0] * max(1, _prod(cards)))
        for idx, asg in enumerate(self._assignments()):
            rest = asg[:k] + asg[k + 1:]
            j = sum(a * st for a, st in zip(rest, out.strides))
            out.values[j] += self.values[idx]
        return out

    def reduce(self, evidence: Dict[str, int]) -> "Factor":
        keep = [i for i, v in enumerate(self.vars) if v not in evidence]
        if len(keep) == len(self.vars):
            return self
        vars_ = [self.vars[i] for i in keep]
        cards = [self.cards[i] for i in keep]
        values = []
        for asg in itertools.product(*[range(c) for c in cards]):
            full = [0] * len(self.vars)
            for i, a in zip(keep, asg):
                full[i] = a
            for i, v in enumerate(self.vars):
                if v in evidence:
                    full[i] = evidence[v]
            values.append(self.values[sum(a * st for a, st in zip(full, self.strides))])
        if not vars_:
            values = [values[0]] if values else [self.values[sum(evidence[v] * st for v, st in zip(self.vars, self.strides))]]
        return Factor(vars_, cards, values)

    def normalize(self) -> "Factor":
        z = sum(self.values) or 1.0
        return Factor(self.vars, self.cards, [v / z for v in self.values])


def _prod(xs: Iterable[int]) -> int:
    p = 1
    for x in xs:
        p *= x
    return p


class Node:
    def __init__(self, name: str, states: List[str], parents: List[str], alpha: float = 1.0,
                 backoff: Optional[List[str]] = None, backoff_strength: float = 20.0):
        self.name = name
        self.states = states
        self.parents = parents
        self.alpha = alpha
        self.backoff = backoff
        self.backoff_strength = backoff_strength
        self.cpt: List[float] = []  # rows over parent configs (row-major), each row over states


class BayesianNetwork:
    def __init__(self) -> None:
        self.nodes: Dict[str, Node] = {}
        self.order: List[str] = []

    def add_node(self, name: str, states: List[str], parents: Optional[List[str]] = None, **kw) -> None:
        parents = parents or []
        for p in parents:
            if p not in self.nodes:
                raise ValueError("parent %s must be added before %s" % (p, name))
        self.nodes[name] = Node(name, states, parents, **kw)
        self.order.append(name)

    def card(self, name: str) -> int:
        return len(self.nodes[name].states)

    def state_index(self, name: str, state) -> int:
        if isinstance(state, int):
            return state
        return self.nodes[name].states.index(state)

    # ------------------------------------------------------------------ learning
    def fit_counts(self, variables: List[str], rows: Iterable[Sequence[int]]) -> None:
        """Learn every CPT in one pass over fully observed rows of state indices
        (columns in the order of `variables`)."""
        pos = {v: i for i, v in enumerate(variables)}
        counts = {}
        back_counts = {}
        for name in self.order:
            node = self.nodes[name]
            counts[name] = [0.0] * (_prod(self.card(p) for p in node.parents) * self.card(name))
            if node.backoff is not None:
                back_counts[name] = [0.0] * (_prod(self.card(p) for p in node.backoff) * self.card(name))
        fams = {}
        for name in self.order:
            node = self.nodes[name]
            fam = [pos[p] for p in node.parents] + [pos[name]]
            cards = [self.card(p) for p in node.parents] + [self.card(name)]
            fams[name] = (fam, _strides(cards))
            if node.backoff is not None:
                bfam = [pos[p] for p in node.backoff] + [pos[name]]
                bcards = [self.card(p) for p in node.backoff] + [self.card(name)]
                fams[name + "#b"] = (bfam, _strides(bcards))
        for row in rows:
            for name in self.order:
                fam, st = fams[name]
                counts[name][sum(row[i] * s for i, s in zip(fam, st))] += 1
                if name in back_counts:
                    bfam, bst = fams[name + "#b"]
                    back_counts[name][sum(row[i] * s for i, s in zip(bfam, bst))] += 1
        for name in self.order:
            self._set_cpt_from_counts(name, counts[name], back_counts.get(name))

    def _set_cpt_from_counts(self, name: str, counts: List[float], back: Optional[List[float]]) -> None:
        node = self.nodes[name]
        k = self.card(name)
        prior_rows = None
        if back is not None:
            prior_rows = []
            for r in range(len(back) // k):
                row = back[r * k:(r + 1) * k]
                z = sum(row) + k
                prior_rows.append([(c + 1.0) / z for c in row])
        cpt = []
        pcards = [self.card(p) for p in node.parents]
        bidx = [node.parents.index(b) for b in node.backoff] if node.backoff is not None else []
        bstr = _strides([self.card(b) for b in node.backoff]) if node.backoff is not None else []
        for r, asg in enumerate(itertools.product(*[range(c) for c in pcards]) if pcards else [()]):
            row = counts[r * k:(r + 1) * k]
            n = sum(row)
            if prior_rows is not None:
                prior = prior_rows[sum(asg[i] * s for i, s in zip(bidx, bstr))]
                a = node.backoff_strength
                cpt.extend((c + a * p) / (n + a) for c, p in zip(row, prior))
            else:
                cpt.extend((c + node.alpha) / (n + node.alpha * k) for c in row)
        node.cpt = cpt

    def set_cpt(self, name: str, table: List[List[float]]) -> None:
        self.nodes[name].cpt = [v for row in table for v in row]

    def cpt_row(self, name: str, parent_states: Sequence[int]) -> List[float]:
        node = self.nodes[name]
        k = self.card(name)
        st = _strides([self.card(p) for p in node.parents])
        r = sum(a * s for a, s in zip(parent_states, st))
        return node.cpt[r * k:(r + 1) * k]

    def factor(self, name: str) -> Factor:
        node = self.nodes[name]
        return Factor(node.parents + [name], [self.card(p) for p in node.parents] + [self.card(name)], node.cpt)

    # ----------------------------------------------------------------- inference
    def query(self, target: str, evidence: Optional[Dict[str, object]] = None,
              soft_evidence: Optional[Dict[str, Sequence[float]]] = None) -> List[float]:
        """P(target | evidence) by variable elimination. Evidence values may be
        state names or indices; soft evidence gives a likelihood per state."""
        ev = {k: self.state_index(k, v) for k, v in (evidence or {}).items()}
        soft = dict(soft_evidence or {})
        if target in ev:
            out = [0.0] * self.card(target)
            out[ev[target]] = 1.0
            return out
        # fast path: all parents observed and nothing below matters
        node = self.nodes[target]
        if all(p in ev for p in node.parents) and target not in soft and not self._has_observed_descendant(target, ev, soft):
            return list(self.cpt_row(target, [ev[p] for p in node.parents]))
        relevant = self._ancestors({target} | set(ev) | set(soft))
        factors = [self.factor(n).reduce(ev) for n in self.order if n in relevant]
        for var, lik in soft.items():
            if var in relevant and var not in ev:
                factors.append(Factor([var], [self.card(var)], list(lik)))
        hidden = [n for n in self.order if n in relevant and n != target and n not in ev]
        while hidden:
            # min-size heuristic: eliminate the variable whose product factor is smallest
            def cost(v: str) -> int:
                vs = set()
                for f in factors:
                    if v in f.vars:
                        vs.update(f.vars)
                return _prod(self.card(x) for x in vs)
            var = min(hidden, key=cost)
            hidden.remove(var)
            involved = [f for f in factors if var in f.vars]
            factors = [f for f in factors if var not in f.vars]
            if not involved:
                continue
            prod = involved[0]
            for f in involved[1:]:
                prod = prod.product(f)
            factors.append(prod.marginalize(var))
        result = factors[0]
        for f in factors[1:]:
            result = result.product(f)
        result = result.normalize()
        if result.vars != [target]:
            for v in list(result.vars):
                if v != target:
                    result = result.marginalize(v)
            result = result.normalize()
        return result.values

    def _ancestors(self, names: Iterable[str]) -> set:
        out = set()
        stack = list(names)
        while stack:
            n = stack.pop()
            if n in out:
                continue
            out.add(n)
            stack.extend(self.nodes[n].parents)
        return out

    def _has_observed_descendant(self, name: str, ev: Dict[str, int], soft: Dict[str, Sequence[float]]) -> bool:
        children = {n for n in self.order if name in self.nodes[n].parents}
        stack = list(children)
        seen = set()
        while stack:
            c = stack.pop()
            if c in seen:
                continue
            seen.add(c)
            if c in ev or c in soft:
                return True
            stack.extend(n for n in self.order if c in self.nodes[n].parents)
        return False

    # ----------------------------------------------------------------- persistence
    def to_dict(self) -> Dict[str, object]:
        return {"nodes": [{"name": n, "states": self.nodes[n].states, "parents": self.nodes[n].parents,
                           "cpt": [round(v, 6) for v in self.nodes[n].cpt]} for n in self.order]}

    @classmethod
    def from_dict(cls, d: Dict[str, object]) -> "BayesianNetwork":
        bn = cls()
        for nd in d["nodes"]:  # type: ignore[index]
            bn.add_node(nd["name"], nd["states"], nd["parents"])
            bn.nodes[nd["name"]].cpt = nd["cpt"]
        return bn

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(self.to_dict(), fh)

    @classmethod
    def load(cls, path: str) -> "BayesianNetwork":
        with open(path, "r", encoding="utf-8") as fh:
            return cls.from_dict(json.load(fh))


def _strides(cards: Sequence[int]) -> List[int]:
    out = []
    s = 1
    for c in reversed(list(cards)):
        out.insert(0, s)
        s *= c
    return out
