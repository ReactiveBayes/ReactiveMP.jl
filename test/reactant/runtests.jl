# Compiling inference with Reactant: a linear Gaussian state-space model, unrolled over a few steps,
# built and run with ReactiveMP inside a function that Reactant compiles. Run with
# `make test-reactant`; it is not part of the root suite, since Reactant is a heavy dependency.

using Test, LinearAlgebra
using ReactiveMP, Reactant, BayesBase, MessagePassingRulesBase, Rocket
import ReactiveMP: activate!, get_stream_of_marginals, RandomVariableActivationOptions, DataVariableActivationOptions, FactorNodeActivationOptions, MessageProductContext

# Reactant's traced numbers are not `Real`, so the distributions here put no bound on their number type.

# A multivariate normal by its mean `μ` and covariance `Σ`.
struct TracedMvNormal{M <: AbstractVector, V <: AbstractMatrix}
    μ::M
    Σ::V
end

# A multivariate normal by its weighted mean `ξ = Λμ` and precision `Λ`.
struct TracedMvNormalWeighted{X <: AbstractVector, W <: AbstractMatrix}
    ξ::X
    Λ::W
end

BayesBase.mean(d::TracedMvNormal) = d.μ
BayesBase.cov(d::TracedMvNormal) = d.Σ

BayesBase.default_prod_rule(::Type{<:TracedMvNormal}, ::Type{<:TracedMvNormalWeighted}) = BayesBase.ClosedProd()
BayesBase.default_prod_rule(::Type{<:TracedMvNormalWeighted}, ::Type{<:TracedMvNormal}) = BayesBase.ClosedProd()
BayesBase.default_prod_rule(::Type{<:TracedMvNormalWeighted}, ::Type{<:TracedMvNormalWeighted}) = BayesBase.ClosedProd()

function BayesBase.prod(::BayesBase.ClosedProd, prior::TracedMvNormal, likelihood::TracedMvNormalWeighted)
    F = lu(I + prior.Σ * likelihood.Λ)
    return TracedMvNormal(F \ (prior.μ + prior.Σ * likelihood.ξ), F \ prior.Σ)
end
BayesBase.prod(strategy::BayesBase.ClosedProd, likelihood::TracedMvNormalWeighted, prior::TracedMvNormal) =
    BayesBase.prod(strategy, prior, likelihood)
BayesBase.prod(::BayesBase.ClosedProd, left::TracedMvNormalWeighted, right::TracedMvNormalWeighted) =
    TracedMvNormalWeighted(left.ξ + right.ξ, left.Λ + right.Λ)

# `out ~ TracedMvNormal(μ, Σ)`, the prior.
@define_factor_node(node = TracedMvNormal, type = Stochastic, interfaces = [:out, :μ, :Σ])

@define_message_update_rule(
    node = TracedMvNormal,
    target = :out,
    args = (q[:μ]::PointMass, q[:Σ]::PointMass),
    body = (args) -> TracedMvNormal(mean(args.q[:μ]), mean(args.q[:Σ])),
)

# `out ~ N(A in, Σ)`: a transition, or an observation of `in`.
struct LinearGaussian end

@define_factor_node(node = LinearGaussian, type = Stochastic, interfaces = [:out, :in, :A, :Σ])

@define_message_update_rule(
    node = LinearGaussian,
    target = :out,
    args = (m[:in]::TracedMvNormal, q[:A]::PointMass, q[:Σ]::PointMass),
    body = (args) -> begin
        A, belief = mean(args.q[:A]), args.m[:in]
        TracedMvNormal(A * belief.μ, A * belief.Σ * A' + mean(args.q[:Σ]))
    end,
)

@define_message_update_rule(
    node = LinearGaussian,
    target = :in,
    args = (q[:out]::PointMass, q[:A]::PointMass, q[:Σ]::PointMass),
    body = (args) -> begin
        A, observation, noise = mean(args.q[:A]), mean(args.q[:out]), lu(mean(args.q[:Σ]))
        TracedMvNormalWeighted(A' * (noise \ observation), A' * (noise \ A))
    end,
)

@define_message_update_rule(
    node = LinearGaussian,
    target = :in,
    args = (m[:out]::TracedMvNormalWeighted, q[:A]::PointMass, q[:Σ]::PointMass),
    body = (args) -> begin
        A, Σ, message = mean(args.q[:A]), mean(args.q[:Σ]), args.m[:out]
        F = lu(I + message.Λ * Σ)
        TracedMvNormalWeighted(A' * (F \ message.ξ), A' * (F \ message.Λ) * A)
    end,
)

# Builds the model over the observations `ys`, runs it once and returns the posterior means of the
# states, as RxInfer would: variables first, then the nodes, then the subscriptions and the data.
function smoothed_means(ys, m0, V0, A, B, P, Q)
    x0 = randomvar()
    xs = [randomvar() for _ in ys]
    observations = [datavar() for _ in ys]
    nodes = Any[factornode(TracedMvNormal, [(:out, x0), (:μ, constvar(m0)), (:Σ, constvar(V0))], ((:out,), (:μ,), (:Σ,)))]
    previous = x0
    for (x, y) in zip(xs, observations)
        push!(nodes, factornode(LinearGaussian, [(:out, x), (:in, previous), (:A, constvar(A)), (:Σ, constvar(P))], ((:out, :in), (:A,), (:Σ,))))
        push!(nodes, factornode(LinearGaussian, [(:out, y), (:in, x), (:A, constvar(B)), (:Σ, constvar(Q))], ((:out,), (:in,), (:A,), (:Σ,))))
        previous = x
    end

    product = MessageProductContext()
    foreach(x -> activate!(x, RandomVariableActivationOptions(nothing, product, product)), [x0; xs])
    foreach(y -> activate!(y, DataVariableActivationOptions(true, false, nothing, nothing)), observations)
    foreach(node -> activate!(node, FactorNodeActivationOptions()), nodes)

    posteriors = Vector{Any}(undef, length(xs))
    subscriptions = [subscribe!(get_stream_of_marginals(x), (q) -> (posteriors[i] = getdata(q))) for (i, x) in enumerate(xs)]
    foreach(new_observation!, observations, ys)
    foreach(unsubscribe!, subscriptions)
    return map(mean, posteriors)
end

θ = π / 35
A = [cos(θ) -sin(θ); sin(θ) cos(θ)]
B, P, Q = Matrix(1.0I, 2, 2), Matrix(1.0I, 2, 2), Matrix(25.0I, 2, 2)
m0, V0 = zeros(2), Matrix(100.0I, 2, 2)
ys = [[cos(t / 10), sin(t / 10)] .* 10 for t in 1:5]

@testset "ReactiveMP compiled with Reactant" begin
    @testset "the extension is loaded" begin
        @test Base.get_extension(ReactiveMP, :ReactiveMPReactantExt) !== nothing
    end

    @testset "an unrolled model compiles and gives the same posteriors" begin
        # Compiled first: `factornode` caches how it built a node of each shape and builds the next
        # one from that plan, so a run before the compilation would spare the trace the code that
        # builds a node from scratch, which is what the extension is for.
        traced = (map(Reactant.to_rarray, ys), Reactant.to_rarray.((m0, V0, A, B, P, Q))...)
        compiled = @compile smoothed_means(traced...)
        computed = compiled(traced...)
        expected = smoothed_means(ys, m0, V0, A, B, P, Q)
        @test length(computed) == length(expected)
        @test all(isapprox.(Array.(computed), expected; atol = 1.0e-10))
    end
end

struct GroupedInputs end
@define_factor_node(node = GroupedInputs, type = Deterministic, interfaces = [:out, :in...])

function scale_by_group_length(x)
    node = factornode(GroupedInputs, [(:out, randomvar()), ((:in, 1), randomvar()), ((:in, 2), randomvar())])
    return x .* ReactiveMP.group_length(node, :in)
end

@testset "group length under Reactant tracing" begin
    x = Reactant.to_rarray([3.0])
    compiled = @compile scale_by_group_length(x)
    @test Array(compiled(x)) == [6.0]
    # Reuse the compiled function: the input must still be traced.
    @test Array(compiled(Reactant.to_rarray([5.0]))) == [10.0]
end
