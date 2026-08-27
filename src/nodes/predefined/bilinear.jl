export Bilinear

@doc raw"""
    Bilinear

Stochastic factor node representing the bilinear interaction

```math
\phi(\mathrm{out}, \mathrm{in}, a) = \exp(\mathrm{out} \cdot a \cdot \mathrm{in})
```

This is the pairwise (edge) potential of Gaussian Belief Propagation as formulated by
Shental et al., *Gaussian Belief Propagation for Solving Systems of Linear Equations*
([arXiv:0810.1119](https://arxiv.org/abs/0810.1119)), whose edge potential
``\psi_{ij}(x_i, x_j) = \exp(-x_i A_{ij} x_j)`` corresponds to ``a = -A_{ij}``. Combined with
`NormalWeightedMeanPrecision(b_i, A_{ii})` self-potentials it turns the solution of
``A x = b`` into message passing on a graph.

# Interfaces
1. `out` — first interaction variable.
2. `in` — second interaction variable.
3. `a` — coupling coefficient, must be a `PointMass`.

# Factorization

Only the structured factorization `q(out, in) q(a)` is supported:

```julia
@constraints begin
    q(out, in, a) = q(out, in)q(a)
end
```

When `a` is supplied as a constant — the intended usage, as in
`x[j] ~ Bilinear(x[i], -A[i, j])` — this factorization is applied automatically, because
constant interfaces are always factorized out of the local cluster. `MeanField()` is **not**
supported: there are no `q(out)q(in)` message rules and no mean-field average energy.

# Improperness

The potential is not integrable on its own, so it is not a conditional distribution:

- the messages towards `out` and `in` are improper Gaussians with negative precision. This is
  inherent to the factor and reproduces the GaBP message
  ``P_{ij} = -A_{ij}^2 / P_{i \setminus j}``;
- the joint marginal `q(out, in)` is proper only when the incoming precisions dominate the
  coupling, ``w_{\mathrm{out}} w_{\mathrm{in}} > a^2``. This is a *local* condition on a single
  factor and must not be confused with the convergence conditions below.

The Bethe free energy is still meaningful whenever the product of all factors in the model is
normalizable. Since ``\langle -\log \phi \rangle = -\mathbb{E}[a]\, \mathbb{E}[\mathrm{out} \cdot
\mathrm{in}]`` carries no normalizer term, the accumulated score equals
``-\log Z + \mathrm{KL}(q \Vert p)``.

# Accuracy

Per Shental et al.:

- if `A` is strictly diagonally dominant, or the diagonally normalized system satisfies
  ``\rho(|I - A|) < 1``, GaBP converges and the inferred **means are exact**, that is, they
  solve ``A x = b``;
- the inferred **variances are exact only on acyclic graphs**. On graphs with cycles they are
  walk-sum approximations of ``\mathrm{diag}(A^{-1})`` and must not be reported as the marginal
  variances of the exact solution.
"""
struct Bilinear end

@node Bilinear Stochastic [out, in, a]

@average_energy Bilinear (q_out_in::Any, q_a::PointMass) = begin
    # ⟨-log φ⟩ = -E[a] ⋅ E_{q(out, in)}[out ⋅ in] = -E[a] ⋅ (V[1, 2] + m[1] ⋅ m[2])
    m, V = mean_cov(q_out_in)
    return -mean(q_a) * (V[1, 2] + m[1] * m[2])
end
