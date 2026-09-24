"""
    GaussianCouplingMessagePassingRules

The GaussianCoupling node, `φ(out, in, a) = exp(out ⋅ a ⋅ in)`, the pairwise potential of Gaussian
belief propagation, and its rules under the structured factorisation `q(out, in) q(a)`.
"""
module GaussianCouplingMessagePassingRules

using MessagePassingRulesBase, BayesBase, ExponentialFamily, Distributions

export GaussianCoupling

@doc raw"""
    GaussianCoupling

A stochastic factor node representing the pairwise Gaussian coupling (a bilinear interaction)

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

`((:out, :in), (:a,))` in `factornode`, as RxInfer's
`@constraints begin q(out, in, a) = q(out, in)q(a) end`.

When `a` is supplied as a constant — the intended usage, as in
`x[j] ~ GaussianCoupling(x[i], -A[i, j])` — this factorization is applied automatically, because
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
struct GaussianCoupling end

@define_factor_node(node = GaussianCoupling, type = Stochastic, interfaces = [:out, :in, :a])

# ⟨-log φ⟩ = -E[a] E[out ⋅ in] = -E[a] (V[1, 2] + m[1] m[2]), with no normaliser.
@define_average_energy(
    node = GaussianCoupling, args = (q[:out, :in]::Any, q[:a]::PointMass),
    body = (args) -> begin
        m, V = mean_cov(args.q[:out, :in])
        -mean(args.q[:a]) * (V[1, 2] + m[1] * m[2])
    end,
)

# m(in) ∝ ∫ exp(a ⋅ out ⋅ in) m_out(out) d(out) = exp(a μ_out in + a² v_out in² / 2): improper,
# with a negative precision, as the factor is; and symmetrically towards `out`.
coupled_message(a, m) = NormalWeightedMeanPrecision(a * mean(m), -abs2(a) * var(m))

@define_message_update_rule(
    node = GaussianCoupling, target = :in, args = (m[:out]::UnivariateNormalDistributionsFamily, q[:a]::PointMass),
    body = (args) -> coupled_message(mean(args.q[:a]), args.m[:out]),
)

@define_message_update_rule(
    node = GaussianCoupling, target = :out, args = (m[:in]::UnivariateNormalDistributionsFamily, q[:a]::PointMass),
    body = (args) -> coupled_message(mean(args.q[:a]), args.m[:in]),
)

# q(out, in) ∝ m_out(out) m_in(in) exp(a ⋅ out ⋅ in): the factor adds the cross term only; proper
# when precision(m_out) ⋅ precision(m_in) > a².
@define_marginal_update_rule(
    node = GaussianCoupling, target = (:out, :in),
    args = (m[:out]::UnivariateNormalDistributionsFamily, m[:in]::UnivariateNormalDistributionsFamily, q[:a]::PointMass),
    body = (args) -> begin
        a = mean(args.q[:a])
        ξ = [weightedmean(args.m[:out]), weightedmean(args.m[:in])]
        W = [precision(args.m[:out]) -a; -a precision(args.m[:in])]
        MvNormalWeightedMeanPrecision(ξ, W)
    end,
)

end
