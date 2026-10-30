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

import cmath
import itertools
import math
import numbers
import os
from argparse import ArgumentTypeError
from collections.abc import Callable, Sequence
from enum import IntEnum
from fractions import Fraction
from typing import TYPE_CHECKING, Any, Literal, SupportsFloat, TypeVar

from .symbolic import Poly

if TYPE_CHECKING:
    # Imported here to prevent circularity
    from .graph.base import ET, VT, BaseGraph

FloatInt = float | int
FractionLike = Fraction | int | Poly

def scale_phase(phase: FractionLike, factor: FractionLike) -> FractionLike:
    """Return ``factor * r`` (mod 2), where ``r`` is the value of ``phase`` reduced into [0, 2).

    Formulas like ``e^{i pi p1 p2}`` or ``3/2 * a - 1/2`` assume the phase is in [0, 2). For a
    symbolic phase like ``c + 1`` (``c`` Boolean) that is not the case: at ``c = 1`` it is 2, and
    ``factor * 2`` is not 0 mod 2 when ``factor`` is fractional. If ``phase`` is a constant plus a
    Pauli polynomial, the Pauli part is rewritten as its parity (``x xor y = x + y - 2xy``) so
    that the product is correct for every assignment. Otherwise this is just ``factor * phase``.
    """
    if not isinstance(phase, Poly):
        return factor * (phase % 2)
    const = sum((c for c, t in phase.terms if not t.vars), Fraction(0))
    pauli = phase - const
    if isinstance(const, complex) or not isinstance(pauli, Poly) or not pauli.is_pauli:
        return factor * phase
    c0 = Fraction(const) % 2
    bits = [Poly([(1, t)]) for c, t in pauli.terms if t.vars and not isinstance(c, complex) and c % 2 == 1]

    # Terms of order k in the parity expansion carry a factor 2^(k-1), so they vanish mod 2
    # once 2^(k-2) * factor is an integer. Only expand up to that order.
    max_order = len(bits)
    coeffs = [c for c, _ in factor.terms] if isinstance(factor, Poly) else [factor]
    real_coeffs = [c for c in coeffs if not isinstance(c, complex)]
    if len(real_coeffs) == len(coeffs):
        dens = [Fraction(c).denominator for c in real_coeffs]
        if all(d & (d - 1) == 0 for d in dens):
            max_order = min(max_order, max((d.bit_length() for d in dens), default=0))

    # r = c0 + parity if c0 < 1, else c0 - parity
    sign = 1 if c0 < 1 else -1
    result: FractionLike = factor * c0
    for k in range(1, max_order + 1):
        for subset in itertools.combinations(bits, k):
            mono = subset[0]
            for b in subset[1:]:
                mono = mono * b
            result = result + (sign * (-2) ** (k - 1)) * (factor * mono)
    return result


def push_pauli_axel(axel: FractionLike, leaf: FractionLike) -> FractionLike:
    """Return the new ``leaf`` phase acquired when a Pauli ``axel`` is pushed
    through a spider of the opposite type. Used by
    :func:`pyzx.rewrite_rules.push_pauli_rule.unsafe_pauli_push`.

    The caller must ensure ``axel`` evaluates to a Pauli phase. The result is
    undefined for non-Pauli axels.
    """
    return (leaf - 2 * scale_phase(axel, leaf)) % 2


def assert_phase_real(phase: FractionLike | complex) -> None:
    """Raise a TypeError if ``phase`` contains complex coefficients.

    Phases represent multiples of pi and must be real-valued.  Complex
    coefficients are only valid in Z-box labels, which are stored as
    vertex data rather than as phases.
    """
    if isinstance(phase, complex) and phase.imag != 0:
        raise TypeError(f"Phase must be real, got {phase}")
    if isinstance(phase, Poly):
        for c, _ in phase.terms:
            if isinstance(c, complex) and c.imag != 0:
                raise TypeError(
                    f"Phase must have real coefficients, got {c}")


def normalize_phase(phase: Any) -> FractionLike:
    """Coerce a phase to a supported type (``int``, ``Fraction``, or ``Poly``).

    When ``settings.strict_phase_types`` is ``True`` (the default), a float-valued
    phase raises ``TypeError`` rather than being silently rounded. Set
    ``settings.strict_phase_types`` to ``False`` to opt in to automatic
    conversion to ``Fraction`` using
    ``settings.float_to_fraction_max_denominator``, accepting the resulting
    precision loss. Other types are returned unchanged.

    Both built-in ``float`` and non-integer numpy reals such as ``numpy.float32``
    are treated as float-valued; numpy's ``float64`` already inherits from
    ``float`` and is caught by the built-in branch.
    """
    is_float_like = isinstance(phase, float) or (
        isinstance(phase, numbers.Real)
        and not isinstance(phase, (int, Fraction, numbers.Integral))
    )
    if is_float_like:
        if settings.strict_phase_types:
            raise TypeError(
                f"Float phase {phase!r} is not accepted by default. Float "
                "phases would be silently rounded to a nearby fraction; "
                "if that is acceptable, opt in by setting "
                "`pyzx.settings.strict_phase_types = False` (the resulting "
                "fraction uses `pyzx.settings.float_to_fraction_max_denominator`)."
            )
        return Fraction(float(phase)).limit_denominator(
            settings.float_to_fraction_max_denominator)
    return phase

def half_phase(phase: FractionLike) -> FractionLike:
    return Fraction(phase / 2) if isinstance(phase, int) else phase / 2

class VertexType(IntEnum):
    """Type of a vertex in the graph."""
    BOUNDARY = 0
    Z = 1
    X = 2
    H_BOX = 3
    W_INPUT = 4
    W_OUTPUT = 5
    Z_BOX = 6
    DUMMY = 99

def vertex_is_zx(ty: VertexType) -> bool:
    """Check if a vertex type corresponds to a green or red spider."""
    return ty in (VertexType.Z, VertexType.X)

def toggle_vertex(ty: VertexType) -> VertexType:
    """Swap the X and Z vertex types."""
    if not vertex_is_zx(ty):
        return ty
    return VertexType.Z if ty == VertexType.X else VertexType.X

def vertex_is_z_like(ty: VertexType) -> bool:
    """Check if a vertex type corresponds to a Z spider or Z box."""
    return ty == VertexType.Z or ty == VertexType.Z_BOX

def vertex_is_zx_like(ty: VertexType) -> bool:
    """Check if a vertex type corresponds to a Z or X spider or Z box."""
    return vertex_is_z_like(ty) or ty == VertexType.X

def vertex_is_w(ty: VertexType) -> bool:
    return ty == VertexType.W_INPUT or ty == VertexType.W_OUTPUT

def get_w_partner(g: 'BaseGraph[VT, ET]', v: 'VT') -> 'VT':
    assert vertex_is_w(g.type(v))
    for edge in g.incident_edges(v):
        if g.edge_type(edge) == EdgeType.W_IO:
            s, t = g.edge_st(edge)
            u = s if t == v else t
            assert vertex_is_w(g.type(u))
            return u
    assert False

def get_w_io(g: 'BaseGraph[VT, ET]', v: 'VT') -> tuple['VT', 'VT']:
    v2 = get_w_partner(g, v)
    if g.type(v) == VertexType.W_INPUT:
        return v, v2
    return v2, v


class EdgeType(IntEnum):
    """Type of an edge in the graph."""
    SIMPLE = 1
    HADAMARD = 2
    W_IO = 3

def toggle_edge(ty: EdgeType) -> EdgeType:
    """Swap the regular and Hadamard edge types."""
    return EdgeType.HADAMARD if ty == EdgeType.SIMPLE else EdgeType.SIMPLE

def phase_to_s(a: FractionLike, t: VertexType = VertexType.Z, poly_with_pi: bool = False, limit_denominator: bool = True) -> str:
    if isinstance(a, Fraction) or isinstance(a, int):
        return phase_fraction_to_s(a, t, limit_denominator=limit_denominator)
    else: # a is a Poly
        if poly_with_pi:
            return f"({a})\u03c0"
        else:
            return str(a)

def phase_fraction_to_s(a: FractionLike, t: VertexType = VertexType.Z, limit_denominator: bool = True) -> str:
    if (a == 0 and t != VertexType.H_BOX): return ''
    if (a == 1 and t == VertexType.H_BOX): return ''
    if isinstance(a, Poly):
        return str(a)
    a = Fraction(a)

    if a == 0: return '0'
    simstr = ''
    if limit_denominator and a.denominator > 256:
        a = a.limit_denominator(256)
        simstr = '~'

    ns = '' if a.numerator == 1 else str(a.numerator)
    ds = '' if a.denominator == 1 else '/' + str(a.denominator)

    # unicode 0x03c0 = pi
    return simstr + ns + '\u03c0' + ds

def phase_is_clifford(phase: FractionLike) -> bool:
    if isinstance(phase, Poly):
        return phase.is_clifford
    if isinstance(phase, (Fraction, int)):
        return phase in [Fraction(i, 2) for i in range(4)]
    raise TypeError(f"phase must be FractionLike, got {type(phase).__name__}")

def phase_is_t_like(phase: FractionLike) -> bool:
    """Whether ``phase`` is an odd multiple of pi/4 for every assignment of its (Boolean)
    variables, i.e. non-Clifford but Clifford once 1/4 is removed (e.g. 3/4, or a + 1/4)."""
    return not phase_is_clifford(phase) and phase_is_clifford(phase - Fraction(1, 4))

def phase_is_proper_clifford(phase: FractionLike) -> bool:
    """Whether ``phase`` is pi/2 or 3pi/2 for every assignment of its (Boolean) variables,
    i.e. it is 1/2 or 3/2 plus a Pauli polynomial such as ``a + 1/2``."""
    if isinstance(phase, Poly):
        const = sum((c for c, t in phase.terms if not t.vars), Fraction(0))
        if isinstance(const, complex) or const % 2 not in (Fraction(1, 2), Fraction(3, 2)):
            return False
        pauli = phase - const
        return not isinstance(pauli, Poly) or pauli.is_pauli
    if isinstance(phase, (Fraction, int)):
        return phase in (Fraction(1, 2), Fraction(3, 2))
    raise TypeError(f"phase must be FractionLike, got {type(phase).__name__}")

def phase_is_pauli(phase: FractionLike) -> bool:
    if isinstance(phase, Poly):
        return phase.is_pauli
    if isinstance(phase, (Fraction, int)):
        return phase in (0, 1)
    raise TypeError(f"phase must be FractionLike, got {type(phase).__name__}")

tikz_classes = {
    'boundary': 'none',
    'Z': 'Z dot',
    'X': 'X dot',
    'Z phase': 'Z phase dot',
    'X phase': 'X phase dot',
    'Z box': 'Z box',
    'H': 'hadamard',
    'W': 'W triangle',
    'W input': 'W input',
    'dummy': 'label',
    'edge': '',
    'H-edge': 'hadamard edge',
    'W-io-edge': 'W io edge'
}

original_colors = {
    'edge': '#000000',
    'Hedge': '#0088ff',
    'Xedge': '#999999',
    'boundary': '#000000',
    'X': '#ff8888',
    'Y': '#aabbff',
    'Z': '#ccffcc',
    'H': '#ffff66',
    'W': '#000000',
    'Zalt': '#ccffcc',
    'Walt': '#000000',
    'Xdark': '#ff8888',
    'Ydark': '#aabbff',
    'Zdark': '#99dd99',
}

rgb_colors = original_colors.copy()
rgb_colors['Y'] = original_colors['Z']
rgb_colors['Z'] = original_colors['Y']
rgb_colors['Ydark'] = original_colors['Zdark']
rgb_colors['Zdark'] = original_colors['Ydark']
rgb_colors['Hedge'] = '#ff6600'

grayscale_colors = {
    'edge': '#000000',
    'Hedge': '#888888',
    'Xedge': '#dddddd',
    'boundary': '#000000',
    'X': '#666666',
    'Y': '#9999dd',
    'Z': '#dddddd',
    'H': '#eeeeee',
    'W': '#000000',
    'Zalt': '#dddddd',
    'Walt': '#000000',
    'Xdark': '#666666',
    'Ydark': '#9999dd',
    'Zdark': '#dddddd',
}

class Settings: # namespace class
    mode: Literal["notebook", "browser", "shell", ""] = "shell"
    drawing_backend: Literal["d3","matplotlib"] = "d3"
    # Maximum denominator when converting floats to fractions (e.g., for phase values).
    # Higher values preserve more precision but may produce larger fractions.
    float_to_fraction_max_denominator: int = 2**20
    # When True, reject float phases at `set_phase`/`add_to_phase` rather than
    # silently rounding them to a nearby fraction. Set to False to opt in to
    # automatic conversion using `float_to_fraction_max_denominator`.
    strict_phase_types: bool = True
    drawing_auto_hbox: bool = False
    javascript_location: str = "" # Path to javascript files of pyzx
    d3_load_string: str = ""
    tikzit_location: str = "" # Path to tikzit executable
    quantomatic_location: str = "" # Path to quantomatic executable
    topt_command: list[str] | None = None # Argument-separated command to run TOpt such as ["wsl", "./TOpt"]
    show_labels: bool = False
    tikz_classes: dict[str, str] = tikz_classes
    default_qasm_version: int = 2
    colors: dict[str, str] = original_colors
    javascript_importmap: dict[str, Any] = {
        "imports": {
            "three": "https://cdn.jsdelivr.net/npm/three@0.172.0/build/three.module.js",
            "three/addons/": "https://cdn.jsdelivr.net/npm/three@0.172.0/examples/jsm/",
            "d3": "https://cdn.jsdelivr.net/npm/d3@5.16.0/dist/d3.min.js",
        }
    }

settings = Settings()


settings.javascript_location = os.path.join(os.path.dirname(__file__), 'js')

# We default to importing d3 from a CDN
settings.d3_load_string = 'require.config({paths: {d3: "https://cdnjs.cloudflare.com/ajax/libs/d3/5.16.0/d3.min"} });'
# However, if we are working in the pyzx directory itself, we can use the copy of d3
# local to pyzx, which doesn't require an internet connection
# We only do this if we believe we are running in the PyZX directory itself.

try:
    relpath = os.path.relpath(settings.javascript_location, os.getcwd())
    if relpath.count('..') <= 2: # We are *probably* working in the PyZX directory
        settings.javascript_location = os.path.relpath(settings.javascript_location, os.getcwd())
        #settings.d3_load_string = 'require.config({{baseUrl: "{}",paths: {{d3: "d3.v5.min"}} }});'.format(
        #                    settings.javascript_location.replace('\\','/'))
        # TODO: This will fail if Jupyter is started in the parent directory of pyzx, while
        # the notebook is not in the pyzx directory
except ValueError: # relpath raises this Exception when the drive letters don't match
    pass


settings.mode = ''
def get_mode() -> Literal['', 'notebook', 'browser', 'shell']:
    if settings.mode == '':
        try:
            import IPython  # type: ignore
            ipython_instance = IPython.get_ipython()
            if ipython_instance is None: raise Exception
            if 'IPKernelApp' in ipython_instance.config:
                settings.mode = "notebook"
                if os.environ.get('SPHINXTARGET') == 'latexpdf':
                    settings.drawing_backend = "matplotlib"
            ipython_instance.config.InlineBackend.figure_format = 'svg'
        except:
            try:
                import browser  # type: ignore
                settings.mode = "browser"
            except:
                settings.mode = "shell"

    return settings.mode



def restricted_float(x: str | SupportsFloat) -> float:
    x = float(x)
    if x < 0.0 or x > 1.0:
        raise ArgumentTypeError("%r not in range [0.0, 1.0]." % (x,))
    return x


def make_into_list(possible_list: Any) -> list[Any]:
    if not isinstance(possible_list, list):
        return [possible_list]
    return possible_list

T = TypeVar('T')
def maxelements(seq: Sequence[T], key: Callable[[T], Any] | None = None, reverse: bool = False) -> list[int]:
    """
    Return list of position(s) of largest element.

    Adapted from: https://stackoverflow.com/questions/3989016/how-to-find-all-positions-of-the-maximum-value-in-a-list
    """
    indices = []
    if key is None:
        key = lambda x: x
    if reverse:
        compare = lambda x, y: x <= y
    else:
        compare = lambda x, y: x >= y
    if seq:
        best_val = key(seq[0])
        for i, val in enumerate(seq):
            cur_val = key(val)
            if compare(cur_val, best_val):
                if cur_val == best_val:
                    indices.append(i)
                else:
                    best_val = cur_val
                    indices = [i]

    return indices


def is_pauli(phase: Any) -> bool:
    """
    Check whether phase is Pauli.

    Compatible with zxlive symbols.
    """
    if phase == 0 or phase == 1:
        return True
    return getattr(phase, 'is_pauli', False)

def get_z_box_label(g: 'BaseGraph[VT, ET]', v: 'VT') -> Any:
    assert g.type(v) == VertexType.Z_BOX
    return g.vdata(v, 'label', 1)

def set_z_box_label(g: 'BaseGraph[VT, ET]', v: 'VT', label: Any) -> None:
    assert g.type(v) == VertexType.Z_BOX
    g.set_vdata(v, 'label', label)


def get_h_box_label(g: 'BaseGraph[VT, ET]', v: 'VT') -> complex:
    assert g.type(v) == VertexType.H_BOX
    label = g.vdata(v, 'label', None)
    if label is not None:
        return complex(label)
    phase = g.phase(v)
    if isinstance(phase, Poly):
        raise ValueError("Cannot convert symbolic phase to complex label")
    return cmath.exp(1j * math.pi * float(phase))

def set_h_box_label(g: 'BaseGraph[VT, ET]', v: 'VT', label: complex) -> None:
    assert g.type(v) == VertexType.H_BOX
    g.set_vdata(v, 'label', complex(label))
    g.set_phase(v, 0)

def is_standard_hbox(g: 'BaseGraph[VT, ET]', v: 'VT') -> bool:
    """Check if H-box has the standard Hadamard label (-1)."""
    assert g.type(v) == VertexType.H_BOX
    label = g.vdata(v, 'label', None)
    if label is not None:
        return cmath.isclose(label, -1)
    return g.phase(v) == 1

def hbox_has_complex_label(g: 'BaseGraph[VT, ET]', v: 'VT') -> bool:
    """Check if H-box uses a complex label instead of legacy phase."""
    assert g.type(v) == VertexType.H_BOX
    return g.vdata(v, 'label', None) is not None


# Return position 'perc'%-distance between 2 points:
def ave_pos(a: float, b: float, perc: float = 1/2) -> float:
    return (abs(a - b))*(perc) + min(a, b)
