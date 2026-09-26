@doc """
    ConjugateAR

The autoregressive node with its parameters on one edge: the likelihood of [`AR`](@ref), but
with the coefficients and the precision joint on the edge `w = (θ, γ)`, a normal-gamma:

```math
p(y \\mid x, w) = \\mathcal{N}\\big(y_1 \\mid \\theta^\\top x, \\, \\gamma^{-1}\\big), \\qquad w = (\\theta, \\gamma).
```

$(DOC_AR_STATE)

Keeping `(θ, γ)` joint makes the parameter update conjugate: under `q(y, x) q(w)` the message
towards `w` is a normal-gamma, and so is `q(w)` for a normal-gamma prior, the posterior of a
Bayesian linear regression with unknown noise precision.

# Interfaces

- `y` (alias `out`): the current state, a multivariate normal;
- `x`: the previous state, a multivariate normal of the same dimension;
- `w`: the coefficients and the precision, an `MvNormalGamma` of dimension `p`.

$(DOC_AR_ALGORITHM) Its form must be `Multivariate`, an AR(1) included: every rule reads `q(w)`,
whose `θ` is a vector.

The rules towards `y` and `x` and the joint marginal `q(y, x)` are [`AR`](@ref)'s, computed
from the marginals of `θ` and `γ` that `q(w)` implies. The message towards `w` needs the
structured factorisation `q(y, x) q(w)`; it is improper for `p ≥ 3`, its shape `(3 - p)/2`
being non-positive, and becomes proper in its product with the prior. There is an average
energy under `q(y, x) q(w)` only.

# Examples

```jldoctest
julia> result = @call_message_update_rule(
           node = ConjugateAR, target = :w, algorithm = ARVMP(Multivariate, 1, ARsafe()),
           clusters = ((:y, :x) => MvNormalMeanCovariance([1.0, 0.5], [1.0 0.0; 0.0 1.0]),),
       );

julia> getresult(result) isa MvNormalGamma
true
```

See also [`AR`](@ref), [`ARVMP`](@ref).
"""
struct ConjugateAR end

@define_factor_node(node = ConjugateAR, type = Stochastic, interfaces = [(:y, aliases = [:out]), :x, :w])

# The marginals of θ and γ under which AR's algebra reproduces ConjugateAR's moments, for
# q(w) = MvNormalGamma(μ, Λ, α, β): ⟨γ⟩ = α / β, ⟨θ⟩ = μ and ⟨γ⟩ Vθ = Λ⁻¹, the coupling
# ⟨γθθᵀ⟩ - ⟨γ⟩μμᵀ.
function conjugatear_effective_marginals(q_w::MvNormalGamma)
    μ, Λ, α, β = params(q_w)
    mγ = α / β
    q_θ = MvNormalMeanCovariance(μ, cholinv(Λ) / mγ)
    q_γ = GammaShapeRate(α, β)
    return q_θ, q_γ
end

@define_message_update_rule(
    node = ConjugateAR, target = :y, algorithm = ARVMP,
    args = (m[:x]::NormalDistributionsFamily, q[:w]::MvNormalGamma),
    body = (algo, args) -> ar_towards_y(algo, args.m[:x], conjugatear_effective_marginals(args.q[:w])...),
)

@define_message_update_rule(
    node = ConjugateAR, target = :y, algorithm = ARVMP,
    args = (q[:x]::Any, q[:w]::MvNormalGamma),
    body = (algo, args) -> ar_towards_y_meanfield(algo, args.q[:x], conjugatear_effective_marginals(args.q[:w])...),
)

@define_message_update_rule(
    node = ConjugateAR, target = :x, algorithm = ARVMP,
    args = (m[:y]::NormalDistributionsFamily, q[:w]::MvNormalGamma),
    body = (algo, args) -> ar_towards_x(algo, args.m[:y], conjugatear_effective_marginals(args.q[:w])...),
)

@define_message_update_rule(
    node = ConjugateAR, target = :x, algorithm = ARVMP,
    args = (q[:y]::Any, q[:w]::MvNormalGamma),
    body = (algo, args) -> ar_towards_x_meanfield(algo, args.q[:y], conjugatear_effective_marginals(args.q[:w])...),
)

# Towards `w`: the AR likelihood as a function of (θ, γ), averaged over q(y, x), a Normal-Gamma
# factor with natural parameters (b, -C/2, 1/2, -a/2) for
#
#     C = ⟨x xᵀ⟩ = Vx + mx mxᵀ,   b = ⟨x y₁⟩ = cov(x, y₁) + mx my₁,   a = ⟨y₁²⟩ = Vy₁ + my₁²
#
# and in mean parameters Λ = C, μ = C⁻¹b, α = (3 - d)/2, β = (a - bᵀC⁻¹b)/2. It is improper for
# order ≥ 3 (α ≤ 0); its product with a prior, the marginal q(w), is the proper MvNormalGamma
# posterior, since natural parameters add. There is no marginal rule over `w` alone: the engine
# forms a single interface's marginal from its messages.
@define_message_update_rule(
    node = ConjugateAR, target = :w, algorithm = ARVMP,
    args = (q[:y, :x]::MultivariateNormalDistributionsFamily,),
    body = (algo, args) -> begin
        order = getorder(algo)

        myx, Vyx = mean_cov(args.q[:y, :x])
        x_range = (order + 1):(2order)

        mx = myx[x_range]
        my1 = first(myx)
        Vx = Vyx[x_range, x_range]
        Vy1 = Vyx[1, 1]
        cov_x_y1 = Vyx[x_range, 1]

        C = Vx + mx * transpose(mx)
        b = cov_x_y1 + mx * my1
        a = Vy1 + my1^2

        μ = cholinv(C) * b
        α = convert(eltype(μ), (3 - order) / 2)
        β = (a - dot(b, μ)) / 2

        MvNormalGamma(μ, C, α, β)
    end,
)

@define_marginal_update_rule(
    node = ConjugateAR, target = (:y, :x), algorithm = ARVMP,
    args = (m[:y]::NormalDistributionsFamily, m[:x]::NormalDistributionsFamily, q[:w]::MvNormalGamma),
    body = (algo, args) -> ar_joint(algo, args.m[:y], args.m[:x], conjugatear_effective_marginals(args.q[:w])...),
)

@define_average_energy(
    node = ConjugateAR, algorithm = ARVMP,
    args = (q[:y, :x]::MultivariateNormalDistributionsFamily, q[:w]::MvNormalGamma),
    body = (algo, args) -> ar_energy(algo, args.q[:y, :x], conjugatear_effective_marginals(args.q[:w])...),
)
