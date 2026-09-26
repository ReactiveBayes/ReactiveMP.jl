@doc raw"""
    ContinuousTransition

The stochastic node of a linear Gaussian transition whose matrix is itself learned:

```math
p(y \mid x, a, W) = \mathcal{N}\big(y \mid A x,\; W^{-1}\big), \qquad A = f(a),
```

taking an `m`-dimensional `x` to an `n`-dimensional `y` through the `n × m` matrix `f(a)`. The
transformation `f` is carried by the node's algorithm, [`CTVMP`](@ref), which the model must
give. [`CTransition`](@ref) is an alias.

# Interfaces

- `y`: the output, an `n`-dimensional vector;
- `x`: the input, an `m`-dimensional vector;
- `a`: the parameters of the matrix, always a vector, of length one for a single parameter;
- `W`: the `n × n` precision of the transition noise.

# Rules

Every rule is variational and runs under [`CTVMP`](@ref), in one of two factorisations:

- structured, `q(y, x) q(a) q(W)`: messages towards `y` and `x` from the messages on `x` and
  `y` (`MvNormal`), messages towards `a` and `W` and the average energy from the joint
  `q(y, x)` (`MvNormal`), and the marginal rule for `q(y, x)`;
- mean-field, `q(y) q(x) q(a) q(W)`: every message and the average energy from the marginals.

`q(a)` must have a mean and a covariance, `q(W)` a mean and, for the average energy, the
expectation of `logdet`, as a Wishart has. The rule towards `a` also reads `q(a)`, its
expansion point, besides the inputs its factorisation gives it, so `q(a)` needs an initial
marginal.

# Linearisation

The rules treat each row of `A` as linear in `a`. With `Fᵢ` the Jacobian of the `i`-th row of `f`
at the mean `mₐ` of `q(a)`, and `Ā` the matrix they use for `f(mₐ)`,

```math
A(a) \approx O + \begin{bmatrix} (F_1 a)^\top \\ \vdots \\ (F_n a)^\top \end{bmatrix},
\qquad O = \bar{A} - \begin{bmatrix} (F_1 m_a)^\top \\ \vdots \\ (F_n m_a)^\top \end{bmatrix}.
```

`Ā` is `f` linearised at `mₐ + σₐ`, the mean plus one standard deviation of `q(a)` per
component, and evaluated at `mₐ`, which is `f(mₐ)` exactly for a linear `f`. The offset `O` is
zero for an `f` linear through the origin, such as `reshape`, and the rules towards `a` and `W`
and the average energy keep it for any other `f`, such as a rotation. The expectation of `AᵀWA`
then has the closed form `ĀᵀW̄Ā + Σᵢⱼ W̄ᵢⱼ Fᵢ Σₐ Fⱼᵀ`, `W̄` being the mean of `q(W)` and `Σₐ`
the covariance of `q(a)`, which every rule uses.

The message towards `a` is a normal with weighted mean and precision

```math
\xi = \sum_i F_i^\top \, \mathbb{E}\big[x (y - O x)^\top\big] \, \bar{W}_{:, i}, \qquad
\Lambda = \sum_i F_i^\top \, \mathbb{E}[x x^\top] \sum_j \bar{W}_{j i} F_j,
```

and the message towards `W` is a Wishart with `n + 2` degrees of freedom and the inverse scale
`E[(y - A x)(y - A x)ᵀ]`.

# Limitations

- Variational only: under the default algorithm,
  [`DefaultAlgorithm`](@extref MessagePassingRulesBase.DefaultAlgorithm), no rule exists, and a
  model that does not name [`CTVMP`](@ref) finds none.
- Multivariate normal messages and marginals on `y`, `x` and `a` only.
- A nonlinear `f` is linearised as above, so its rules are approximate.

# Examples

A 2 × 2 transition learned entry by entry, with `a` known almost exactly to be the identity:

```jldoctest
julia> using ContinuousTransitionMessagePassingRules, MessagePassingRulesBase, ExponentialFamily, BayesBase, LinearAlgebra

julia> algorithm = CTVMP(a -> reshape(a, 2, 2));

julia> result = @call_message_update_rule(
           node = ContinuousTransition, target = :y, algorithm = algorithm,
           m = (x = MvNormalMeanCovariance([1.0, 2.0], [1.0 0.0; 0.0 1.0]),),
           q = (a = MvNormalMeanCovariance([1.0, 0.0, 0.0, 1.0], 1e-8 * Matrix(1.0I, 4, 4)), W = Wishart(3, [1.0 0.0; 0.0 1.0])),
       );

julia> m, V = mean_cov(getresult(result));

julia> m ≈ [1.0, 2.0] && V ≈ (1 + 1 / 3) * I
true
```

The covariance is `A Σₓ Aᵀ` plus the inverse of the mean of `q(W)`, `3 I`.

See also [`CTVMP`](@ref), [`CTransition`](@ref).
"""
struct ContinuousTransition end

"""
    CTransition

An alias for [`ContinuousTransition`](@ref).
"""
const CTransition = ContinuousTransition

"""
    CTVMP(f)

The algorithm of [`ContinuousTransition`](@ref), carrying the transformation `f` that takes the
vector `a` to the matrix `A = f(a)`. The node declares no algorithm of its own, so a model must
give this one; its rules are variational.

# Arguments

- `f`: a function of a vector returning an `n × m` matrix, differentiable by ForwardDiff. The
  rules differentiate it at the mean of `q(a)`, and a nonlinear `f` is linearised there.

# Examples

```jldoctest
julia> using ContinuousTransitionMessagePassingRules

julia> CTVMP(a -> reshape(a, 2, 2)).f([1.0, 2.0, 3.0, 4.0])
2×2 Matrix{Float64}:
 1.0  3.0
 2.0  4.0
```

A rotation by one angle, a single parameter:

```julia
CTVMP(a -> [cos(a[1]) -sin(a[1]); sin(a[1]) cos(a[1])])
```
"""
struct CTVMP{F} <: AbstractAlgorithm
    f::F
end

gettransformation(algo::CTVMP) = algo.f

@define_factor_node(node = ContinuousTransition, type = Stochastic, interfaces = [:y, :x, :a, :W])

# Every target follows the factorisation; the one towards `a` also reads `q(a)`, its expansion
# point.
@define_dependencies(
    node = ContinuousTransition, algorithm = CTVMP,
    dependencies = [:y => (default,), :x => (default,), :a => (default, q[:a]), :W => (default,)],
)

# The Jacobians of f's rows at `a`: `Fs[i]` is the `dx × da` derivative of the row `f(a)[i, :]`, so
# that for a linear `f` the row is `(Fs[i] a)ᵀ`. One Jacobian of `vec ∘ f`, whose `i`-th row of `A`
# is the entries `i, i + dy, …`.
function jacobians(algo::CTVMP, a)
    f = gettransformation(algo)
    dy = size(f(a), 1)
    J = ForwardDiff.jacobian(a -> vec(f(a)), a)
    return [J[i:dy:end, :] for i in 1:dy]
end

# `A` for the rules: `f` linearised at `a0 = a + epsilon` and evaluated at `a`, which is `f(a)`
# exactly for a linear `f`. The rules expand at the mean plus a standard deviation of `q(a)`.
function ct_matrix(algo::CTVMP, a, epsilon)
    f = gettransformation(algo)
    a0 = a + epsilon
    A0 = f(a0)
    J0 = ForwardDiff.jacobian(a -> vec(f(a)), a0)
    return A0 + reshape(J0 * (a - a0), size(A0))
end

ct_matrix(algo::CTVMP, q_a) = ct_matrix(algo, mean(q_a), sqrt.(var(q_a)))

# The blocks of a joint q(y, x) over `dy` components of `y`: the means of `y` and `x`, their
# covariances, and the cross-covariance Cov(y, x), `dy × dx`.
function split_y_x(q_y_x, dy)
    m, V = mean_cov(q_y_x)
    y, x = 1:dy, (dy + 1):length(m)
    return m[y], V[y, y], m[x], V[x, x], V[y, x]
end

# The linearised A as an offset plus a part linear in `a`: row `i` is `Āᵢ + (Fᵢ (a - m_a))ᵀ`, so
# `A(a) = O + [(Fᵢ a)ᵀ]ᵢ` with `O = Ā - [(Fᵢ m_a)ᵀ]ᵢ`. O is zero for an `f` linear through the
# origin, such as `reshape`, and the rules towards `a` and `W` keep it for any other `f`.
ct_offset(Ā, Fs, ma) = Ā - reduce(vcat, [(F * ma)' for F in Fs])

# Σⱼ W[j, i] Fs[j] for each `i`: the term the rules and the energy weight the Jacobians by.
weighted_jacobians(mW, Fs) = [sum(mW[j, i] * Fs[j] for j in eachindex(Fs)) for i in eachindex(Fs)]
