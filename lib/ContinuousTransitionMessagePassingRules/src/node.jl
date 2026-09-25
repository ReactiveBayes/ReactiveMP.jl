@doc raw"""
    ContinuousTransition

The node

```math
y \sim \mathcal{N}(A x, W^{-1}), \qquad A = f(a),
```

which takes an `m`-dimensional `x` to an `n`-dimensional `y` through the `n × m` matrix `f(a)`,
built from the vector `a` by the transformation `f` of its algorithm, [`CTVMP`](@ref). With
`f = a -> reshape(a, n, m)` every entry of `A` is learned; a structured `f`, such as a rotation
`a -> [cos(a[1]) -sin(a[1]); sin(a[1]) cos(a[1])]`, learns fewer parameters. `a` is always a
vector, of length one for a single parameter. Also available as `CTransition`.

Its interfaces are `y`, `x`, `a` and `W`, the precision of the transition. Its rules are
variational, under mean-field `q(y) q(x) q(a) q(W)` or the structured `q(y, x) q(a) q(W)`: each
target follows the factorisation, and the rule towards `a` also reads `q(a)`, the point it
expands `f` around. The node declares no algorithm of its own, and under the default one,
`DefaultAlgorithm`, no rule exists.
"""
struct ContinuousTransition end

const CTransition = ContinuousTransition

"""
    CTVMP(f)

The algorithm of [`ContinuousTransition`](@ref), which the model must give: `f` takes the vector
`a` to the matrix `A` and must return a matrix. A nonlinear `f` is linearised with ForwardDiff.

```jldoctest
julia> using ContinuousTransitionMessagePassingRules

julia> CTVMP(a -> reshape(a, 2, 2)).f([1.0, 2.0, 3.0, 4.0])
2×2 Matrix{Float64}:
 1.0  3.0
 2.0  4.0
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
