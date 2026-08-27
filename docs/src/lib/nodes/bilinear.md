# [Bilinear node](@id lib-nodes-bilinear)

The `Bilinear` node encodes the stochastic bilinear interaction

```math
\phi(\mathrm{out}, \mathrm{in}, a) = \exp(\mathrm{out} \cdot a \cdot \mathrm{in})
```

This is the pairwise (edge) potential of Gaussian Belief Propagation (GaBP) as formulated by Shental et al., *Gaussian Belief Propagation for Solving Systems of Linear Equations* ([arXiv:0810.1119](https://arxiv.org/abs/0810.1119)). Their edge potential ``\psi_{ij}(x_i, x_j) = \exp(-x_i A_{ij} x_j)`` corresponds to ``a = -A_{ij}``, and their self-potential ``\phi_i(x_i) = \exp(b_i x_i - A_{ii} x_i^2 / 2)`` is `NormalWeightedMeanPrecision(b_i, A[i, i])`. Together they turn the solution of ``A x = b`` into message passing on a graph.

## [Interfaces](@id lib-nodes-bilinear-interfaces)

| Interface | Role |
|-----------|------|
| `out` | First interaction variable |
| `in` | Second interaction variable |
| `a` | Coupling coefficient, must be a `PointMass` |

## [Factorization](@id lib-nodes-bilinear-factorization)

Only the structured factorization `q(out, in) q(a)` is supported:

```julia
@constraints begin
    q(out, in, a) = q(out, in)q(a)
end
```

When `a` is supplied as a constant — the intended usage, as in `x[j] ~ Bilinear(x[i], -A[i, j])` — this factorization is applied automatically, because constant interfaces are always factorized out of the local cluster. `MeanField()` is **not** supported: there are no `q(out)q(in)` message rules and no mean-field average energy.

## [Improperness](@id lib-nodes-bilinear-improperness)

The potential is not integrable on its own, so it is not a conditional distribution. Note that it is exactly the cross-term of a Gaussian density, since ``\mathcal{N}(\mathrm{out}; \mathrm{in}, w^{-1}) \propto \exp(-w \cdot \mathrm{out}^2/2) \exp(w \cdot \mathrm{out} \cdot \mathrm{in}) \exp(-w \cdot \mathrm{in}^2/2)``, with the two quadratic self-terms removed. Consequences:

- The messages towards `out` and `in` are improper Gaussians with negative precision. This is inherent to the factor and reproduces the GaBP message ``P_{ij} = -A_{ij}^2 / P_{i \setminus j}``.
- The joint marginal `q(out, in)` is proper only when the incoming precisions dominate the coupling, ``w_{\mathrm{out}} w_{\mathrm{in}} > a^2``. This is a *local* condition on a single factor; it is **not** the condition under which the method as a whole works — see [Accuracy](@ref lib-nodes-bilinear-accuracy) for that.

The Bethe free energy is still meaningful whenever the product of all factors in the model is normalizable. Since ``\langle -\log \phi \rangle = -\mathbb{E}[a]\, \mathbb{E}[\mathrm{out} \cdot \mathrm{in}]`` carries no normalizer term, the accumulated score equals ``-\log Z + \mathrm{KL}(q \Vert p)``.

## [Accuracy](@id lib-nodes-bilinear-accuracy)

Per Shental et al.:

- If `A` is strictly diagonally dominant, or the diagonally normalized system satisfies
  ``\rho(|I - A|) < 1``, GaBP converges and the inferred **means are exact**, that is, they
  solve ``A x = b``.
- The inferred **variances are exact only on acyclic graphs**. On graphs with cycles they are
  walk-sum approximations of ``\mathrm{diag}(A^{-1})`` and must not be reported as the
  marginal variances of the exact solution.

```@docs
ReactiveMP.Bilinear
```
