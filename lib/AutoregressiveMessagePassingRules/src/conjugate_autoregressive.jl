@doc raw"""
    ConjugateAR

The autoregressive node with its parameters on one edge: the likelihood of [`AR`](@ref),
`y₁ ~ N(θᵀx, γ⁻¹)` with the companion-form state, but with `(θ, γ)` joint on the edge `w`,
an `MvNormalGamma`, where [`AR`](@ref) has an edge for each.

Keeping `(θ, γ)` joint makes the parameter update conjugate: under `q(y, x) q(w)` the posterior
of `w` is an `MvNormalGamma`, that of a Bayesian linear regression with unknown noise
precision whose statistics are the expected sufficient statistics of `q(y, x)`.

Its interfaces are `y` (alias `out`), `x` and `w`. Its rules run under [`ARVMP`](@ref), as
[`AR`](@ref)'s: the node declares no algorithm of its own, and under the default one,
`DefaultAlgorithm`, no rule exists.
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
