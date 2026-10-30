# PyZX - Python library for quantum circuit rewriting
#        and optimization using the ZX-calculus
# Copyright (C) 2018 - Aleks Kissinger and John van de Wetering

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#    http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import itertools
import unittest
from fractions import Fraction

from pyzx.graph import Graph
from pyzx.graph.scalar import simplify_poly
from pyzx.utils import EdgeType, VertexType, phase_is_proper_clifford, phase_is_t_like, scale_phase
from pyzx.symbolic import Poly, new_var
from pyzx.simplify import gadget_simp, teleport_reduce
from pyzx.rewrite_rules.lcomp_rule import check_lcomp, unsafe_lcomp
from pyzx.rewrite_rules.merge_phase_gadget_rule import merge_phase_gadgets_for_simp
from pyzx.rewrite_rules.push_pauli_rule import unsafe_pauli_push
from pyzx.rewrite_rules.pivot_rule import check_pivot, unsafe_pivot
from pyzx.tensor import compare_tensors


def tensors_match(g1, g2, **subs):
    """True iff `g1` and `g2` give the same tensor after substituting `subs`."""
    return compare_tensors(
        g1.substitute_variables(subs),
        g2.substitute_variables(subs),
        preserve_scalar=True,
    )


class TestBooleanPauliRewrites(unittest.TestCase):

    def test_gadget_simp(self):
        """`gadget_simp` skips a boolean-axel gadget by default; the
        opt-in ``apply_to_boolean_axels=True`` path runs the rewrite and
        preserves the tensor under each substitution of the boolean."""
        def build():
            g = Graph()
            c = new_var('c', is_bool=True, registry=g.var_registry)
            b0 = g.add_vertex(VertexType.BOUNDARY, 0, 0)
            b1 = g.add_vertex(VertexType.BOUNDARY, 0, 4)
            t = g.add_vertex(VertexType.Z, 0, 2)
            n = g.add_vertex(VertexType.Z, 1, 2, phase=c)
            v = g.add_vertex(VertexType.Z, 2, 2, phase=Fraction(1, 4))
            g.add_edge((b0, t)); g.add_edge((t, b1))
            g.add_edge((t, n), EdgeType.HADAMARD); g.add_edge((n, v), EdgeType.HADAMARD)
            g.set_inputs((b0,)); g.set_outputs((b1,))
            return g

        # Default path: gadget_simp leaves the boolean-axel gadget alone.
        g_default = build()
        self.assertFalse(gadget_simp(g_default))

        # Opt-in path: apply via the underlying function (gadget_simp cannot
        # forward the flag) and check tensor equivalence under each boolean value.
        g_optin = build()
        orig = g_optin.copy()
        self.assertTrue(merge_phase_gadgets_for_simp(g_optin, apply_to_boolean_axels=True))
        for c_val in (0, 1):
            self.assertTrue(tensors_match(g_optin, orig, c=c_val),
                            f"gadget_simp drifted at c={c_val}")

    def test_teleport_reduce(self):
        """`teleport_reduce` must commute with substitution for a boolean axel.

        Two phase gadgets sharing parity {target}: boolean axel `c` and
        constant pi axel, both with T leaves.
        """
        def build(c_val=None):
            g = Graph()
            c = (new_var('c', is_bool=True, registry=g.var_registry)
                 if c_val is None else Fraction(c_val))
            b0 = g.add_vertex(VertexType.BOUNDARY, 0, 0)
            b1 = g.add_vertex(VertexType.BOUNDARY, 0, 4)
            t = g.add_vertex(VertexType.Z, 0, 2)
            n1 = g.add_vertex(VertexType.Z, 1, 2, phase=c)
            v1 = g.add_vertex(VertexType.Z, 2, 2, phase=Fraction(1, 4))
            n2 = g.add_vertex(VertexType.Z, -1, 2, phase=Fraction(1))
            v2 = g.add_vertex(VertexType.Z, -2, 2, phase=Fraction(1, 4))
            g.add_edge((b0, t)); g.add_edge((t, b1))
            g.add_edge((t, n1), EdgeType.HADAMARD); g.add_edge((n1, v1), EdgeType.HADAMARD)
            g.add_edge((t, n2), EdgeType.HADAMARD); g.add_edge((n2, v2), EdgeType.HADAMARD)
            g.set_inputs((b0,)); g.set_outputs((b1,))
            return g

        g_tel = teleport_reduce(build())
        for c_val in (0, 1):
            ref = teleport_reduce(build(c_val))
            got = g_tel.substitute_variables({'c': Fraction(c_val)})
            self.assertTrue(compare_tensors(ref, got),
                            f"teleport_reduce drifted at c={c_val}")

    def test_unsafe_pauli_push(self):
        """`unsafe_pauli_push` skips a boolean Pauli by default and
        preserves the tensor when opted in via ``apply_to_boolean_axels=True``."""
        def build():
            g = Graph()
            c = new_var('c', is_bool=True, registry=g.var_registry)
            b0 = g.add_vertex(VertexType.BOUNDARY, 0, 0)
            b1 = g.add_vertex(VertexType.BOUNDARY, 0, 4)
            v = g.add_vertex(VertexType.Z, 0, 1, phase=Fraction(1, 4))
            w = g.add_vertex(VertexType.X, 0, 3, phase=c)
            g.add_edge((b0, v)); g.add_edge((v, w)); g.add_edge((w, b1))
            g.set_inputs((b0,)); g.set_outputs((b1,))
            return g, v, w

        # Default path: return False and leave the graph untouched.
        g_default, v, w = build()
        self.assertFalse(unsafe_pauli_push(g_default, v, w))
        self.assertEqual(g_default.phase(v), Fraction(1, 4))

        # Opt-in path: rewrite runs and the resulting tensor matches the
        # original for each substitution of the boolean axel.
        g_optin, v, w = build()
        orig = g_optin.copy()
        self.assertTrue(unsafe_pauli_push(g_optin, v, w, apply_to_boolean_axels=True))
        # v's phase should now depend on c (a Poly), not be the original Fraction.
        self.assertIsInstance(g_optin.phase(v), Poly)
        for c_val in (0, 1):
            self.assertTrue(tensors_match(g_optin, orig, c=c_val),
                            f"unsafe_pauli_push drifted at c={c_val}")

    def test_unsafe_pivot(self):
        """`unsafe_pivot` must preserve the tensor for boolean pivot phases."""
        g = Graph()
        c1 = new_var('c1', is_bool=True, registry=g.var_registry)
        c2 = new_var('c2', is_bool=True, registry=g.var_registry)
        b0 = g.add_vertex(VertexType.BOUNDARY, 0, 0)
        b1 = g.add_vertex(VertexType.BOUNDARY, 0, 4)
        v0 = g.add_vertex(VertexType.Z, 0, 2, phase=c1)
        v1 = g.add_vertex(VertexType.Z, 0, 2.5, phase=c2)
        g.add_edge((b0, v0), EdgeType.HADAMARD)
        g.add_edge((v0, v1), EdgeType.HADAMARD)
        g.add_edge((v1, b1), EdgeType.HADAMARD)
        g.set_inputs((b0,)); g.set_outputs((b1,))

        orig = g.copy()
        unsafe_pivot(g, v0, v1)
        for c1v in (0, 1):
            for c2v in (0, 1):
                self.assertTrue(tensors_match(g, orig, c1=c1v, c2=c2v),
                                f"unsafe_pivot drifted at c1={c1v}, c2={c2v}")

    def test_unsafe_pauli_push_multi_variable_axel(self):
        """Pushing a Pauli axel ``c + d`` through a T spider: at c = d = 1 the axel is 2, so its
        product with the fractional leaf phase must be taken mod 2 first."""
        g = Graph()
        c = new_var('c', is_bool=True, registry=g.var_registry)
        d = new_var('d', is_bool=True, registry=g.var_registry)
        b0 = g.add_vertex(VertexType.BOUNDARY, 0, 0)
        b1 = g.add_vertex(VertexType.BOUNDARY, 0, 4)
        v = g.add_vertex(VertexType.Z, 0, 1, phase=Fraction(1, 4))
        w = g.add_vertex(VertexType.X, 0, 3, phase=c + d)
        g.add_edge((b0, v)); g.add_edge((v, w)); g.add_edge((w, b1))
        g.set_inputs((b0,)); g.set_outputs((b1,))
        orig = g.copy()
        self.assertTrue(unsafe_pauli_push(g, v, w, apply_to_boolean_axels=True))
        for cv, dv in itertools.product((0, 1), repeat=2):
            self.assertTrue(tensors_match(g, orig, c=cv, d=dv),
                            f"unsafe_pauli_push drifted at c={cv}, d={dv}")

    def _closed_graph(self, centre_phase, leaf_phases):
        """A closed diagram: a Z spider joined by Hadamard edges to a triangle of Z spiders."""
        g = Graph()
        centre = g.add_vertex(VertexType.Z, 0, 0, phase=centre_phase)
        leaves = [g.add_vertex(VertexType.Z, 1, i, phase=p) for i, p in enumerate(leaf_phases)]
        for i, u in enumerate(leaves):
            g.add_edge((centre, u), EdgeType.HADAMARD)
            g.add_edge((u, leaves[(i + 1) % len(leaves)]), EdgeType.HADAMARD)
        return g, centre, leaves

    def test_lcomp_symbolic_clifford(self):
        """lcomp applies to a symbolic +-pi/2 phase in a closed diagram (and preserves the value),
        but not in an open one, where symbolic spiders are left in place."""
        c = new_var('c', is_bool=True)
        d = new_var('d', is_bool=True)
        g, centre, _ = self._closed_graph(Fraction(1, 2) + c + d, [Fraction(1, 4), Fraction(3, 4), c])
        self.assertTrue(check_lcomp(g, centre))
        orig = g.copy()
        unsafe_lcomp(g, centre)
        for cv, dv in itertools.product((0, 1), repeat=2):
            self.assertTrue(tensors_match(g, orig, c=cv, d=dv),
                            f"unsafe_lcomp drifted at c={cv}, d={dv}")

        g_open, centre, leaves = self._closed_graph(Fraction(1, 2) + c, [Fraction(1, 4)] * 3)
        b = g_open.add_vertex(VertexType.BOUNDARY, 2, 0)
        g_open.add_edge((leaves[0], b))
        g_open.set_outputs((b,))
        self.assertFalse(check_lcomp(g_open, centre))

    def test_pivot_symbolic_pauli(self):
        """pivot applies to two symbolic Pauli phases in a closed diagram (and preserves the value),
        but not in an open one."""
        c = new_var('c', is_bool=True)
        d = new_var('d', is_bool=True)
        e = new_var('e', is_bool=True)
        g, centre, leaves = self._closed_graph(c + d + 1, [d + e, Fraction(1, 4), Fraction(3, 4)])
        self.assertTrue(check_pivot(g, centre, leaves[0]))
        orig = g.copy()
        unsafe_pivot(g, centre, leaves[0])
        for cv, dv, ev in itertools.product((0, 1), repeat=3):
            self.assertTrue(tensors_match(g, orig, c=cv, d=dv, e=ev),
                            f"unsafe_pivot drifted at c={cv}, d={dv}, e={ev}")

        g_open, centre, leaves = self._closed_graph(c, [d, Fraction(1, 4), Fraction(3, 4)])
        b = g_open.add_vertex(VertexType.BOUNDARY, 2, 0)
        g_open.add_edge((leaves[1], b))
        g_open.set_outputs((b,))
        self.assertFalse(check_pivot(g_open, centre, leaves[0]))


class TestBooleanPhaseHelpers(unittest.TestCase):

    def test_empty_poly_is_falsy(self):
        """A Poly whose terms cancelled is 0, so ``if phase:`` must treat it as falsy."""
        c = new_var('c', is_bool=True)
        self.assertFalse((c + 1) - (c + 1))
        self.assertTrue(c)
        self.assertTrue(c + 1)

    def test_scale_phase(self):
        """``scale_phase(p, f)`` equals ``f * (p mod 2)`` for every assignment, also when the
        Pauli part of ``p`` can make it leave [0, 2)."""
        a, b, c = (new_var(n, is_bool=True) for n in 'abc')
        phases = [a + 1, a + b, a + b + c, Fraction(3, 2) + a + b + c, Fraction(1, 2) + a]
        factors = [Fraction(1, 4), Fraction(3, 2), Fraction(7, 4), Fraction(5, 4) + c]
        def value(q, vals):
            if not isinstance(q, Poly):
                return Fraction(q)
            return Fraction(simplify_poly(q.substitute({v: vals[v.name] for v in q.free_vars()})))
        for p, f in itertools.product(phases, factors):
            scaled = scale_phase(p, f)
            for bits in itertools.product((0, 1), repeat=3):
                vals = dict(zip('abc', bits))
                self.assertEqual(value(scaled, vals) % 2, (value(f, vals) * (value(p, vals) % 2)) % 2,
                                 f"scale_phase({p}, {f}) at {vals}")

    def test_phase_classifiers(self):
        """``phase_is_t_like`` and ``phase_is_proper_clifford`` hold for every assignment."""
        c = new_var('c', is_bool=True)
        x = new_var('x', is_bool=False)
        self.assertTrue(phase_is_t_like(Fraction(3, 4)))
        self.assertTrue(phase_is_t_like(c + Fraction(1, 4)))
        self.assertFalse(phase_is_t_like(Fraction(1, 2)))
        self.assertFalse(phase_is_t_like(x + Fraction(1, 4)))
        self.assertTrue(phase_is_proper_clifford(Fraction(3, 2)))
        self.assertTrue(phase_is_proper_clifford(c + Fraction(1, 2)))
        self.assertFalse(phase_is_proper_clifford(c))
        self.assertFalse(phase_is_proper_clifford(Fraction(1, 2) * c))


if __name__ == '__main__':
    unittest.main()
