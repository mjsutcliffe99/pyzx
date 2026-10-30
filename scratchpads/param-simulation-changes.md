# Simulating parameterised ZX-diagrams: changes and justification

This change makes PyZX's stabiliser-decomposition simulators (`simulation.simulate`, Strategy `BSS` and `MAGIC_CAT`) work on diagrams with **Boolean parameters**, as in the parametric rewriting approach of [arXiv:2403.06777](https://arxiv.org/abs/2403.06777): a diagram is reduced once, with its inputs left as Boolean variables, and the resulting sum of scalars is evaluated for any assignment.

Before this change, simulating such a diagram crashed. Once the crashes were fixed, the results were wrong for many assignments, and some of these bugs affect concrete diagrams too. Every fix below comes with a regression test that fails on the previous code. The accompanying notebook, `param-simulation-changes.ipynb`, demonstrates each one.

## Background: why symbolic phases need care

A Boolean parameter `a` takes the value 0 or 1, so a phase `aπ` is either 0 or π. Most rewrite rules work unchanged on such phases, because they only add phases together, and addition is compatible with working mod 2π.

The exception is rules whose scalar multiplies a phase by something, e.g. the spider pair's `e^{iπ·p1·p2}` or local complementation's `e^{iπ(3/2·a − 1/2)}`. These formulas are derived for a phase in `[0, 2)`. That's always true for concrete phases, since PyZX stores them mod 2. But a symbolic phase like `c + 1` takes the value 2 at `c = 1`, and multiplying 2 rather than 0 by a fraction changes the result: `3/2 · 2 = 3 ≡ 1 (mod 2)`, where the right answer is 0.

`%2` on a polynomial can't fix this. It reduces coefficients, not values, and `c + 1` and `1 − c` are the same phase mod 2. Only `1 − c` stays in range, though, and only it gives the right product. Being equal mod 2 doesn't survive multiplication by a non-integer: if `p ≡ p'` then `λp − λp' = 2kλ`, which is only even when `λ` is an integer.

## 1. Simulating parameterised diagrams no longer crashes

| Problem | Fix |
|---|---|
| BSS picked T-spiders with `phase.denominator != 4`, which raises `NotImplementedError` for a `Poly`. | New `utils.phase_is_t_like(p)`: non-Clifford, but Clifford once 1/4 is removed. For concrete phases this is the same test, and it also accepts `a + 1/4`. Used by BSS and MAGIC_CAT. |
| `simulate` summed `term.scalar.to_number()`, which raises for symbolic scalars. | If the terms still contain parameters, `simulate` returns the list of term scalars, whose sum is the amplitude as a function of the parameters. Concrete diagrams still return a `complex`. |
| MAGIC_CAT's fallback matched T-spiders with `phase in (0.25, 0.75, 1.25, 1.75)`, so it never found symbolic ones. It then returned the graph unchanged, and `decompose` recursed on it until `RecursionError`. | It uses `phase_is_t_like`, and raises a clear error instead of returning the graph unchanged. |

## 2. Parameterised terms reduce to pure scalars

After decomposition, many parameterised terms still contained spiders, e.g. a triangle `(a+3/2)—(b+3/2)—(c+1/2)`. Local complementation only matched a phase of exactly `1/2` or `3/2`, and pivoting only `0` or `1`, so these spiders were never removed. `simulate` reads only the scalar of each term, so the leftover spiders were silently dropped and the amplitude came out wrong.

- **lcomp** now also matches a phase that is ±π/2 for every assignment: 1/2 or 3/2 plus a Pauli polynomial (`utils.phase_is_proper_clifford`). Phases that are only *sometimes* Clifford, like `½·a`, are excluded, because no single rewrite is valid for them.
- **pivot** now also matches symbolic Pauli phases (`phase_is_pauli`, which already supported `Poly`).
- **Scope:** both only apply to diagrams without inputs or outputs. In open diagrams, e.g. circuits with mid-circuit measurements, symbolic outcome spiders are meant to stay visible rather than be absorbed (tests from tqec/tqec#528 check this). Scalar diagrams are exactly the case simulation needs.
- **Rule bodies:** I checked that they are valid for symbolic phases. They only add phases to neighbours, plus the scalar products handled in section 3.

## 3. Products of symbolic phases: phase pairs (the paper's pi-pair and half-pi subterms)

### The bug
The rules that multiply phases into the scalar multiplied them out straight away and added the result to the scalar's single phase polynomial. That's correct only if each factor is in `[0, 2)`. For example, the spider pair `(a+1+b+c+d)—H—(e+3/4+f+g+h)`:

|  | resulting scalar | size | correct |
|---|---|---|---|
| before | `√2·exp((a⋅e + 3/4⋅a + a⋅f + … + d⋅h)iπ)` | 25 terms | **16 of 256 assignments** |
| after | `√2·exp([a + 1 + b + c + d]⋅[e + 3/4 + f + g + h]iπ)` | 1 pair | 256 of 256 |

Even when the expansion is correct (Pauli × Pauli, e.g. `(a+b+c+d)—H—(e+f+g+h)`), it has `n·m` terms (16 here, plus 4 for the `+1`) instead of the `n + m` needed to store the two factors.

### The fix
`Scalar` gets a new subterm list, `phasepairs`. Each entry `(c, p1, p2)` stands for the factor

    exp(iπ · c · r(p1) · r(p2)),   r(p) = the value of p reduced into [0, 2)

and `r` is only applied once values are substituted. These are the paper's pi-pair subterms, and its half-pi subterms when `p2 = 1`. Rules call `Scalar.add_phase_pair(p1, p2)` instead of `add_phase(p1 * p2)`. That covers the spider pair (Pauli and Clifford cases), lcomp, pivot, bialgebra, pushing a Pauli and merging phase gadgets.

`add_phase_pair` folds the product straight into the phase when that's exact and doesn't grow anything:

| case | result |
|---|---|
| both factors concrete | `(p1 mod 2)·(p2 mod 2)` in the phase, as before (concrete simulation is unchanged) |
| symbolic factor × integer `k` | `k·p` in the phase: the representative doesn't matter for an integer multiplier |
| a factor with at most one variable, e.g. `a + 1` | reduced exactly to a linear polynomial (`1 − a`) with `utils.scale_phase`, then folded in |
| otherwise | kept as a phase pair |

So simple products like `a·b` still appear in the phase as before; three existing tests rely on that. Pairs are only kept where expanding would be wrong or would grow.

### Supporting changes
- **`Scalar`:** `to_number`, `substitute_variables` (which re-adds each pair, so it folds once concrete), `copy`/`conjugate` (`c → −c`), `mult_with_scalar`, `__eq__`, `__str__`/`to_latex`/`to_unicode`, and JSON (`"phasepairs"`; the coefficient is parsed as a plain `Fraction`, not as a phase, so it isn't reduced mod 2).
- **Elsewhere:** `BaseGraph.rebind_variables_to_registry` and `quimb.to_quimb_tensor` handle the pairs.
- **`push_pauli_axel`:** it computes a spider's new phase, which has to be a single polynomial, so it uses `scale_phase` for the exact reduction.

## 4. MAGIC_CAT bugs (these also affect concrete diagrams)

| Problem | Fix |
|---|---|
| **Wrong amplitudes on unreduced diagrams.** `check_catn` accepted a Z spider with opposite-colour (X) neighbours on plain edges, but `gen_catlike_term` only implements the graph-like form: it builds Z spiders and unfuses a Z-phase π/4 from each neighbour. In a raw circuit, a CNOT's Z spider with X neighbours was decomposed as a cat state, e.g. `cliffordT(5, 200, seed=42)` gave `-0.0019+0.0598j` instead of `-0.1612-0.0728j`. This is fixed in every version I tried and is independent of the parameterised work. | `check_catn` accepts only a Z centre with Z neighbours on Hadamard edges. Unreduced diagrams then go through the magic5 fallback first, which reduces them. |
| **The π-centre case of all cat decompositions (cat3 to cat6) was wrong.** The old code added π to the new central spider and applied an ad-hoc phase correction to one leg. It rarely showed up in concrete use, because `full_reduce` usually removes such centres, but they survive more often with parameters. | A π-centre cat state is the 0-centre one with an X on one leg: `\|+…+⟩ − \|−…−⟩ = X_k(\|+…+⟩ + \|−…−⟩)`. Pushing that X out through the leg's T (`T X = e^{iπ/4} X T†`, `T† = S† T`) gives `cat_π = e^{iπ/4} · X_k · S†_k · cat_0`. Each term is built that way. The existing test now covers π centres too. |

## 5. Smaller fixes found along the way
- **`Poly.__bool__`.** A `Poly` whose terms have cancelled compares equal to 0 but counted as true, so `if phase:` checks treated it as non-zero. For example, the copy rule added a phase of π for a phase-0 spider. This only showed up for some vertex numberings, which is why it was intermittent.
- **`sum_of_phases` keys.** A phase that cancels to an empty `Poly` is now stored as 0. Before, it was a separate key from 0, so the two terms never merged: `{0: 1, Poly(): −1}` should be `{}`. It also printed as `""`, which broke the JSON round trip.

## Compatibility
- **`simulate`'s return value** for a diagram with free variables is now a `list[Scalar]` rather than an exception. Concrete diagrams are unaffected.
- **Code relying on the old behaviour:** `check_catn` rejects the opposite-colour form it used to accept but couldn't decompose correctly. Products of multi-variable symbolic phases are no longer expanded into `Scalar.phase`, so code reading the global phase polynomial directly should also look at `Scalar.phasepairs`.
- **Non-Boolean (continuous) parameters** take the same paths as before. The widened matching only applies to Boolean phases, and the results checked so far are correct (see below). A diagram whose continuous-phase spiders can't be removed by T-decompositions now ends in an explicit error (`No magic states to replace` or `No T-like spiders to decompose, but the graph is not Clifford`) rather than a crash or a wrong value. That's expected, since only T-like spiders are decomposed.

## Testing
- **Full suite:** 635 tests pass, 11 skipped. `mypy pyzx/ tests/` reports one error, in `rank_width.py`, which is pre-existing and unrelated: it also appears without these changes.
- **New tests** (each fails on the previous code):
  - `test_boolean_pauli_rewrites.py`: Pauli push with a multi-variable axel; lcomp and pivot on symbolic phases (applied in closed diagrams, not in open ones); empty `Poly` is falsy; `scale_phase` over all assignments; `phase_is_t_like` and `phase_is_proper_clifford`.
  - `test_scalar.py` (`TestScalarPhasePairs`): multi-variable factors kept factored; the Pauli × T-like example over all 256 assignments; single-variable factors fold; conjugate/copy/JSON of pairs; concrete pairs fold; cancelled `sum_of_phases` keys.
  - `test_simulation.py`: π-centre cat decompositions; MAGIC_CAT on an unreduced circuit; `check_catn` rejecting the opposite-colour form; parameterised BSS and MAGIC_CAT against the concrete circuit for every bitstring, with every term fully reduced.
- **Random sweeps** outside the test suite:
  - parameterised BSS and MAGIC_CAT on 36 random Clifford+T circuits, every bitstring against the exact tensor, plus the concrete circuits: all correct;
  - earlier rounds of 120+ runs under several random and hash seeds;
  - continuous and mixed parameters: `full_reduce` (320 evaluations), spider-pair scalars (150) and `simulate` (60 runs; all returned results correct, 3 explicit errors as described above).

## Known limitations / follow-ups
- **`sum_of_phases`** still multiplies spider-pair sums out, so it grows as 4^k with k generic spider pairs. The paper keeps these as separate spider-pair subterms; that would be the analogous next step.
- **Continuous parameters:** spiders with continuous phases aren't decomposed. If needed, a vertex-cut fallback would handle them, at the cost of doubling the number of terms per cut.
