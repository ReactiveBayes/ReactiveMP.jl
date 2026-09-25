# Cases (a) to (d) of Phase 4.5, belief propagation, variational message passing, a mixture and
# a Delta node,
# compared call by call with the trajectories v6 recorded through RxInfer
# (`compat/v6-comparison/fixtures/engine`).
#
# The case (b) and (c) graphs are built in GraphPPL's order, which is the order RxInfer
# activates them in: variables as the model statements create them, each constant after the
# random variable of its statement, and data where it is first used. Posteriors are subscribed
# in the order RxInfer's `returnvars` `Dict` iterates: `μ, τ, x`, `m, π, p, z` and `z, x`.

@testitem "engine:fixture:bp_iid" tags = [:engine] setup = [EngineHarness] begin
    using ExponentialFamily, StandardMessagePassingRules, MessagePassingRulesTestUtils
    import ReactiveMP: LogScaleAnnotations
    H = EngineHarness

    Y = [1.2, 0.7, 2.1, 1.6, 0.9, 1.4]
    graph = H.Graph()
    y = [H.data!(graph) for _ in Y]
    prior_v = H.data!(graph)
    v = H.data!(graph)
    x = H.random!(graph)
    zero_mean = H.constant!(graph, 0.0)
    H.node!(graph, NormalMeanVariance, [(:out, x), (:μ, zero_mean), (:v, prior_v)])
    for i in eachindex(Y)
        H.node!(graph, NormalMeanVariance, [(:out, y[i]), (:μ, x), (:v, v)])
    end

    trajectory = H.run(
        graph; id = "bp_iid", data = [y => Y, prior_v => 10.0, v => 1.0], iterations = 2,
        posteriors = [:x => x], annotations = (LogScaleAnnotations(),),
    )
    @test compare_engine_trajectory(trajectory, H.fixture("bp_iid"); atol = 1.0e-9) === :agree
end

@testitem "engine:fixture:bp_iid_missing" tags = [:engine] setup = [EngineHarness] begin
    using ExponentialFamily, StandardMessagePassingRules, MessagePassingRulesTestUtils
    H = EngineHarness

    Y = [1.2, missing, 2.1, 1.6, 0.9, 1.4]
    graph = H.Graph()
    y = [H.data!(graph) for _ in Y]
    prior_v = H.data!(graph)
    v = H.data!(graph)
    x = H.random!(graph)
    zero_mean = H.constant!(graph, 0.0)
    H.node!(graph, NormalMeanVariance, [(:out, x), (:μ, zero_mean), (:v, prior_v)])
    for i in eachindex(Y)
        H.node!(graph, NormalMeanVariance, [(:out, y[i]), (:μ, x), (:v, v)])
    end

    trajectory = H.run(
        graph; id = "bp_iid_missing", data = [y => Y, prior_v => 10.0, v => 1.0], iterations = 2,
        posteriors = [:x => x], predictions = y, free_energy = false,
    )
    @test compare_engine_trajectory(trajectory, H.fixture("bp_iid_missing"); atol = 1.0e-9) === :agree
end

@testitem "engine:fixture:bp_chain" tags = [:engine] setup = [EngineHarness] begin
    using ExponentialFamily, StandardMessagePassingRules, MessagePassingRulesTestUtils
    H = EngineHarness

    Y = [1.2, 0.7, 2.1, 1.6, 0.9, 1.4]
    graph = H.Graph()
    y = [H.data!(graph) for _ in Y]
    prior_v = H.data!(graph)
    v = H.data!(graph)
    x = [H.random!(graph) for _ in Y]
    zero_mean = H.constant!(graph, 0.0)
    H.node!(graph, NormalMeanVariance, [(:out, x[1]), (:μ, zero_mean), (:v, prior_v)])
    H.node!(graph, NormalMeanVariance, [(:out, y[1]), (:μ, x[1]), (:v, v)])
    for i in 2:length(Y)
        H.node!(graph, NormalMeanVariance, [(:out, x[i]), (:μ, x[i - 1]), (:v, v)])
        H.node!(graph, NormalMeanVariance, [(:out, y[i]), (:μ, x[i]), (:v, v)])
    end

    trajectory = H.run(
        graph; id = "bp_chain", data = [y => Y, prior_v => 10.0, v => 1.0], iterations = 2,
        posteriors = [:x => x],
    )
    @test compare_engine_trajectory(trajectory, H.fixture("bp_chain"); atol = 1.0e-9) === :agree
end

@testitem "engine:fixture:vmp_meanfield" tags = [:engine] setup = [EngineHarness] begin
    using ExponentialFamily, StandardMessagePassingRules, MessagePassingRulesTestUtils
    H = EngineHarness

    Y = [1.2, 0.7, 2.1, 1.6, 0.9, 1.4]
    graph = H.Graph()
    μ = H.random!(graph)
    μ_prior = [(:out, μ), (:μ, H.constant!(graph, 0.0)), (:τ, H.constant!(graph, 0.01))]
    τ = H.random!(graph)
    τ_prior = [(:out, τ), (:α, H.constant!(graph, 1.0)), (:β, H.constant!(graph, 1.0))]
    y = [H.data!(graph) for _ in Y]
    H.node!(graph, NormalMeanPrecision, μ_prior; factorisation = H.meanfield_factorisation(μ_prior))
    H.node!(graph, GammaShapeRate, τ_prior; factorisation = H.meanfield_factorisation(τ_prior))
    for i in eachindex(Y)
        interfaces = [(:out, y[i]), (:μ, μ), (:τ, τ)]
        H.node!(graph, NormalMeanPrecision, interfaces; factorisation = H.meanfield_factorisation(interfaces))
    end

    trajectory = H.run(
        graph; id = "vmp_meanfield", data = [y => Y], iterations = 5,
        posteriors = [:μ => μ, :τ => τ], initial_marginals = [τ => GammaShapeRate(1.0, 1.0)],
    )
    @test compare_engine_trajectory(trajectory, H.fixture("vmp_meanfield"); atol = 1.0e-9) === :agree
end

@testitem "engine:fixture:vmp_structured" tags = [:engine] setup = [EngineHarness] begin
    using ExponentialFamily, StandardMessagePassingRules, MessagePassingRulesTestUtils
    H = EngineHarness

    Y = [1.2, 0.7, 2.1, 1.6, 0.9, 1.4]
    graph = H.Graph()
    μ = H.random!(graph)
    μ_prior = [(:out, μ), (:μ, H.constant!(graph, 0.0)), (:τ, H.constant!(graph, 0.01))]
    τ = H.random!(graph)
    τ_prior = [(:out, τ), (:α, H.constant!(graph, 1.0)), (:β, H.constant!(graph, 1.0))]
    x, y, observations = [], [], []
    for _ in Y
        push!(x, H.random!(graph))
        push!(y, H.data!(graph))
        push!(observations, [(:out, y[end]), (:μ, x[end]), (:v, H.constant!(graph, 0.5))])
    end
    H.node!(graph, NormalMeanPrecision, μ_prior)
    H.node!(graph, GammaShapeRate, τ_prior)
    for i in eachindex(Y)
        # q(x, μ)q(τ)
        H.node!(graph, NormalMeanPrecision, [(:out, x[i]), (:μ, μ), (:τ, τ)]; factorisation = ((:out, :μ), (:τ,)))
        H.node!(graph, NormalMeanVariance, observations[i])
    end

    trajectory = H.run(
        graph; id = "vmp_structured", data = [y => Y], iterations = 5,
        posteriors = [:μ => μ, :τ => τ, :x => x],
        initial_marginals = [τ => GammaShapeRate(1.0, 1.0), μ => NormalMeanPrecision(0.0, 1.0)],
    )
    @test compare_engine_trajectory(trajectory, H.fixture("vmp_structured"); atol = 1.0e-9) === :agree
end

@testitem "engine:fixture:normal_mixture" tags = [:engine] setup = [EngineHarness] begin
    using ExponentialFamily, StandardMessagePassingRules, MessagePassingRulesTestUtils
    H = EngineHarness

    Y = [-2.1, -1.8, 2.2, 1.9, -2.3, 2.0, 1.7, -1.9]
    graph = H.Graph()
    meanfield!(fform, interfaces; kwargs...) = H.node!(graph, fform, interfaces; factorisation = H.meanfield_factorisation(interfaces), kwargs...)

    π = H.random!(graph)
    priors = Any[(Dirichlet, [(:out, π), (:a, H.constant!(graph, [1.0, 1.0]))])]
    m, p = [], []
    for (μ, τ) in ((-1.0, 0.1), (1.0, 0.1))
        push!(m, H.random!(graph))
        push!(priors, (NormalMeanPrecision, [(:out, m[end]), (:μ, H.constant!(graph, μ)), (:τ, H.constant!(graph, τ))]))
    end
    for _ in 1:2
        push!(p, H.random!(graph))
        push!(priors, (GammaShapeRate, [(:out, p[end]), (:α, H.constant!(graph, 1.0)), (:β, H.constant!(graph, 1.0))]))
    end
    z, y = [], []
    for _ in Y
        push!(z, H.random!(graph))
        push!(y, H.data!(graph))
    end
    foreach(((fform, interfaces),) -> meanfield!(fform, interfaces), priors)
    for i in eachindex(Y)
        meanfield!(Categorical, [(:out, z[i]), (:p, π)])
        components = [((:m, 1), m[1]), ((:m, 2), m[2]), ((:p, 1), p[1]), ((:p, 2), p[2])]
        meanfield!(NormalMixture, [(:out, y[i]), (:switch, z[i]), components...]; algorithm = NormalMixtureVMP())
    end

    trajectory = H.run(
        graph; id = "normal_mixture", data = [y => Y], iterations = 5,
        posteriors = [:m => m, :π => π, :p => p, :z => z],
        initial_marginals = [
            π => Dirichlet([1.0, 1.0]), m[1] => NormalMeanPrecision(-1.0, 0.1), m[2] => NormalMeanPrecision(1.0, 0.1),
            p[1] => GammaShapeRate(1.0, 1.0), p[2] => GammaShapeRate(1.0, 1.0),
        ],
    )
    # v6 subscribed to a group's members last first; the engine subscribes in declaration order.
    # The members do not depend on each other, so this reorders calls inside an iteration and
    # changes no value (`DISCUSSION.md` §3.24).
    @test compare_engine_trajectory(trajectory, H.fixture("normal_mixture"); atol = 1.0e-9, trace_order = :within_iteration) === :agree
end

@testmodule DeltaFunctions begin
    square_plus_one(x) = x^2 + 1.0
    scaled_square_plus(c, x, s) = c * x^2 + s
    cube_minus(x) = x^3 - x
end

@testitem "engine:fixture:delta_unscented" tags = [:engine] setup = [EngineHarness, DeltaFunctions] begin
    using ExponentialFamily, StandardMessagePassingRules, DeltaMessagePassingRules, MessagePassingRulesApproximations, MessagePassingRulesTestUtils
    H = EngineHarness
    f = DeltaFunctions.square_plus_one

    graph = H.Graph()
    x = H.random!(graph)
    prior = [(:out, x), (:μ, H.constant!(graph, 0.5)), (:v, H.constant!(graph, 1.0))]
    z = H.random!(graph)
    y = H.data!(graph)
    H.node!(graph, NormalMeanVariance, prior)
    H.node!(graph, DeltaFn{typeof(f)}, [(:out, z), ((:in, 1), x)]; algorithm = DeltaApproximation(method = Unscented()), nodefn = f)
    H.node!(graph, NormalMeanVariance, [(:out, y), (:μ, z), (:v, H.constant!(graph, 0.1))])

    trajectory = H.run(
        graph; id = "delta_unscented", data = [y => 2.0], iterations = 3,
        posteriors = [:z => z, :x => x], initial_marginals = [z => NormalMeanVariance(1.0, 1.0)],
    )
    @test compare_engine_trajectory(trajectory, H.fixture("delta_unscented"); atol = 1.0e-9) === :agree
end

@testitem "engine:fixture:delta_linearization" tags = [:engine] setup = [EngineHarness, DeltaFunctions] begin
    using ExponentialFamily, StandardMessagePassingRules, DeltaMessagePassingRules, MessagePassingRulesApproximations, MessagePassingRulesTestUtils
    H = EngineHarness
    f = DeltaFunctions.cube_minus

    graph = H.Graph()
    x, z = H.random!(graph), H.random!(graph)
    y = H.data!(graph)
    H.node!(graph, NormalMeanVariance, [(:out, x), (:μ, H.constant!(graph, 0.5)), (:v, H.constant!(graph, 1.0))])
    H.node!(graph, DeltaFn{typeof(f)}, [(:out, z), ((:in, 1), x)]; algorithm = DeltaApproximation(method = Linearization()), nodefn = f)
    H.node!(graph, NormalMeanVariance, [(:out, y), (:μ, z), (:v, H.constant!(graph, 0.1))])

    trajectory = H.run(
        graph; id = "delta_linearization", data = [y => 2.0], iterations = 3,
        posteriors = [:z => z, :x => x], initial_marginals = [z => NormalMeanVariance(1.0, 1.0)],
    )
    @test compare_engine_trajectory(trajectory, H.fixture("delta_linearization"); atol = 1.0e-9) === :agree
end

@testitem "engine:fixture:delta_unscented_static" tags = [:engine] setup = [EngineHarness, DeltaFunctions] begin
    using ExponentialFamily, StandardMessagePassingRules, DeltaMessagePassingRules, MessagePassingRulesApproximations, MessagePassingRulesTestUtils
    H = EngineHarness
    f = DeltaFunctions.scaled_square_plus

    # `z := f(2.0, x, s)`: the constant and the data are folded into the function, so `x` is
    # the node's only input, `(:in, 1)`.
    graph = H.Graph()
    x = H.random!(graph)
    prior = [(:out, x), (:μ, H.constant!(graph, 0.5)), (:v, H.constant!(graph, 1.0))]
    z = H.random!(graph)
    c = H.constant!(graph, 2.0)
    s = H.data!(graph)
    y = H.data!(graph)
    H.node!(graph, NormalMeanVariance, prior)
    H.node!(graph, DeltaFn{typeof(f)}, [(:out, z), ((:in, 1), c), ((:in, 2), x), ((:in, 3), s)]; algorithm = DeltaApproximation(method = Unscented()), nodefn = f)
    H.node!(graph, NormalMeanVariance, [(:out, y), (:μ, z), (:v, H.constant!(graph, 0.1))])

    trajectory = H.run(
        graph; id = "delta_unscented_static", data = [y => 3.0, s => 1.0], iterations = 3,
        posteriors = [:z => z, :x => x], initial_marginals = [z => NormalMeanVariance(1.0, 1.0)],
    )
    @test compare_engine_trajectory(trajectory, H.fixture("delta_unscented_static"); atol = 1.0e-9) === :agree
end

@testitem "engine:fixture:logic_bp" tags = [:engine] setup = [EngineHarness] begin
    # Belief propagation through deterministic nodes under the default scheme: every message
    # out of a logic node reads the messages on its other interfaces (Phase 5, step 4).
    using ExponentialFamily, StandardMessagePassingRules, MessagePassingRulesTestUtils
    H = EngineHarness

    graph = H.Graph()
    x = H.random!(graph)
    p = H.data!(graph)
    y = H.random!(graph)
    y_prior = [(:out, y), (:p, H.constant!(graph, 0.6))]
    z = H.random!(graph)
    n = H.random!(graph)
    w = H.random!(graph)
    w_prior = [(:out, w), (:p, H.constant!(graph, 0.4))]
    o = H.random!(graph)
    v = H.random!(graph)
    v_prior = [(:out, v), (:p, H.constant!(graph, 0.5))]
    i = H.random!(graph)
    i_factor = [(:out, i), (:p, H.constant!(graph, 0.9))]
    H.node!(graph, Bernoulli, [(:out, x), (:p, p)])
    H.node!(graph, Bernoulli, y_prior)
    H.node!(graph, AND, [(:out, z), (:in1, x), (:in2, y)])
    H.node!(graph, NOT, [(:out, n), (:in, z)])
    H.node!(graph, Bernoulli, w_prior)
    H.node!(graph, OR, [(:out, o), (:in1, n), (:in2, w)])
    H.node!(graph, Bernoulli, v_prior)
    H.node!(graph, IMPLY, [(:out, i), (:in1, o), (:in2, v)])
    H.node!(graph, Bernoulli, i_factor)

    trajectory = H.run(
        graph; id = "logic_bp", data = [p => 0.3], iterations = 2,
        posteriors = [:o => o, :w => w, :n => n, :y => y, :v => v, :z => z, :i => i, :x => x],
    )
    @test compare_engine_trajectory(trajectory, H.fixture("logic_bp"); atol = 1.0e-9) === :agree
end

@testitem "engine:fixture:mixture_bp" tags = [:engine] setup = [EngineHarness] begin
    using ExponentialFamily, StandardMessagePassingRules, MessagePassingRulesTestUtils, Distributions
    import ReactiveMP: LogScaleAnnotations
    H = EngineHarness

    # Mixture's rules read their inputs' log scales, through `ann.m`, and its switch rule
    # multiplies messages through `ctx.product`; v6 recorded every call's log scale.
    graph = H.Graph()
    s = H.random!(graph)
    x = [H.random!(graph), H.random!(graph)]
    z = H.random!(graph)
    y = H.data!(graph)
    H.node!(graph, Categorical, [(:out, s), (:p, H.constant!(graph, [0.3, 0.7]))])
    H.node!(graph, NormalMeanVariance, [(:out, x[1]), (:μ, H.constant!(graph, -2.0)), (:v, H.constant!(graph, 1.0))])
    H.node!(graph, NormalMeanVariance, [(:out, x[2]), (:μ, H.constant!(graph, 2.0)), (:v, H.constant!(graph, 1.0))])
    H.node!(graph, Mixture, [(:out, z), (:switch, s), ((:inputs, 1), x[1]), ((:inputs, 2), x[2])])
    H.node!(graph, NormalMeanVariance, [(:out, y), (:μ, z), (:v, H.constant!(graph, 0.5))])

    trajectory = H.run(
        graph; id = "mixture_bp", data = [y => 1.5], iterations = 2,
        posteriors = [:s => s, :x => x], annotations = (LogScaleAnnotations(),), free_energy = false,
    )
    # v6 makes three calls the port does not, all with values the port computes too: it
    # materialises the two priors' deferred messages a second time in iteration 0, once for the
    # posterior of x[k] and once more through the variable's equality chain, and RxInfer computes
    # z's marginal once when the data arrives, which calls Mixture(:out) in iteration 1; nothing
    # in the port asks for that message, whose rule has hand-derived cases of its own. The rest
    # must agree exactly, in order, log scales included.
    v6 = H.fixture("mixture_bp")
    extra = [4, 5, findfirst(r -> (r.node, r.target) == ("Mixture", ":out"), v6.trace)]
    @test all(k -> (v6.trace[k].iteration, v6.trace[k].node, v6.trace[k].target) == (0, "NormalMeanVariance", ":out"), 4:5)
    expected = EngineTrajectory("mixture_bp"; free_energy = v6.free_energy, posteriors = v6.posteriors, trace = v6.trace[setdiff(eachindex(v6.trace), extra)])
    @test compare_engine_trajectory(trajectory, expected; atol = 1.0e-9) === :agree
end

@testitem "engine:fixture:gaussian_coupling" tags = [:engine] setup = [EngineHarness] begin
    # The message towards `c` is improper, with negative precision; the likelihood's precision
    # makes `c`'s marginal proper. The constant coefficient gives the structured factorisation.
    using ExponentialFamily, StandardMessagePassingRules, GaussianCouplingMessagePassingRules, MessagePassingRulesTestUtils
    H = EngineHarness

    graph = H.Graph()
    x, c = H.random!(graph), H.random!(graph)
    y = H.data!(graph)
    H.node!(graph, NormalMeanPrecision, [(:out, x), (:μ, H.constant!(graph, 0.5)), (:τ, H.constant!(graph, 2.0))])
    H.node!(graph, GaussianCoupling, [(:out, c), (:in, x), (:a, H.constant!(graph, 0.5))])
    H.node!(graph, NormalMeanVariance, [(:out, y), (:μ, c), (:v, H.constant!(graph, 1.0))])

    trajectory = H.run(graph; id = "gaussian_coupling", data = [y => 1.5], iterations = 3, posteriors = [:x => x, :c => c])
    # v6 computes `x`'s prior message twice, once for each of its subscribers, where the engine
    # shares one; and it computes GaussianCoupling's two messages, which do not depend on each
    # other, in another order within an iteration. The values agree.
    @test compare_engine_trajectory(trajectory, H.fixture("gaussian_coupling"); atol = 1.0e-9, trace_order = :within_iteration, collapse_repeats = true) === :agree
end

@testitem "engine:fixture:probit_ep" tags = [:engine] setup = [EngineHarness] begin
    # Each Probit's rule towards `w` reads the message on its own edge, which the other Probit's
    # message feeds: they start from the node's default initial message, NMP(0, 100).
    using ExponentialFamily, StandardMessagePassingRules, ProbitMessagePassingRules, MessagePassingRulesTestUtils
    H = EngineHarness

    graph = H.Graph()
    w = H.random!(graph)
    y1, y2 = H.data!(graph), H.data!(graph)
    H.node!(graph, NormalMeanVariance, [(:out, w), (:μ, H.constant!(graph, 0.0)), (:v, H.constant!(graph, 1.0))])
    H.node!(graph, Probit, [(:out, y1), (:in, w)])
    H.node!(graph, Probit, [(:out, y2), (:in, w)])

    trajectory = H.run(graph; id = "probit_ep", data = [y1 => 1.0, y2 => 0.0], iterations = 5, posteriors = [:w => w])
    @test compare_engine_trajectory(trajectory, H.fixture("probit_ep"); atol = 1.0e-9) === :agree
end

@testitem "engine:fixture:gcv_meanfield" tags = [:engine] setup = [EngineHarness] begin
    # Mean-field through GCV, y's variance about x being exp(z - 0.5), and y observed through a
    # narrow normal, so q(y) is a normal and the node's energy applies. The message towards z is an
    # ExponentialLinearQuadratic, which the product at z turns into a normal by cubature.
    using ExponentialFamily, StandardMessagePassingRules, GCVMessagePassingRules, MessagePassingRulesTestUtils
    H = EngineHarness
    meanfield(interfaces) = Tuple((first(i),) for i in interfaces)
    node!(graph, fform, interfaces) = H.node!(graph, fform, interfaces; factorisation = meanfield(interfaces))

    graph = H.Graph()
    x, y, z = H.random!(graph), H.random!(graph), H.random!(graph)
    o = H.data!(graph)
    node!(graph, NormalMeanVariance, [(:out, x), (:μ, H.constant!(graph, 0.5)), (:v, H.constant!(graph, 1.0))])
    node!(graph, NormalMeanVariance, [(:out, z), (:μ, H.constant!(graph, 0.0)), (:v, H.constant!(graph, 1.0))])
    node!(graph, GCV, [(:y, y), (:x, x), (:z, z), (:κ, H.constant!(graph, 1.0)), (:ω, H.constant!(graph, -0.5))])
    node!(graph, NormalMeanVariance, [(:out, o), (:μ, y), (:v, H.constant!(graph, 0.1))])

    trajectory = H.run(
        graph; id = "gcv_meanfield", data = [o => 2.0], iterations = 5, posteriors = [:x => x, :y => y, :z => z],
        initial_marginals = [x => NormalMeanVariance(0.5, 1.0), y => NormalMeanVariance(2.0, 1.0), z => NormalMeanVariance(0.0, 1.0)],
    )
    @test compare_engine_trajectory(trajectory, H.fixture("gcv_meanfield"); atol = 1.0e-9) === :agree
end

@testitem "engine:fixture:softdot_regression" tags = [:engine] setup = [EngineHarness] begin
    # A Bayesian linear regression through SoftDot under mean-field: y[i] ~ N(θ ⋅ X[i], 1/γ).
    using ExponentialFamily, StandardMessagePassingRules, SoftDotMessagePassingRules, MessagePassingRulesTestUtils
    H = EngineHarness
    meanfield(interfaces) = Tuple((first(i),) for i in interfaces)
    node!(graph, fform, interfaces) = H.node!(graph, fform, interfaces; factorisation = meanfield(interfaces))

    graph = H.Graph()
    θ, γ = H.random!(graph), H.random!(graph)
    node!(graph, MvNormalMeanPrecision, [(:out, θ), (:μ, H.constant!(graph, [0.0, 0.0])), (:Λ, H.constant!(graph, [1.0 0.0; 0.0 1.0]))])
    node!(graph, GammaShapeRate, [(:out, γ), (:α, H.constant!(graph, 2.0)), (:β, H.constant!(graph, 1.0))])
    ys, Xs = [H.data!(graph) for _ in 1:3], [H.data!(graph) for _ in 1:3]
    for (y, X) in zip(ys, Xs)
        node!(graph, SoftDot, [(:y, y), (:θ, θ), (:x, X), (:γ, γ)])
    end

    data = [(Xs .=> [[1.0, 0.5], [0.3, -1.0], [2.0, 1.0]])..., (ys .=> [1.2, -0.4, 2.1])...]
    trajectory = H.run(
        # Under mean-field the order of updates is the schedule: v6 updated θ, then γ from the new θ,
        # and in the engine the posterior subscribed last updates first.
        graph; id = "softdot_regression", data, iterations = 5, posteriors = [:γ => γ, :θ => θ],
        initial_marginals = [θ => MvNormalMeanPrecision([0.0, 0.0], [1.0 0.0; 0.0 1.0]), γ => GammaShapeRate(2.0, 1.0)],
    )
    @test compare_engine_trajectory(trajectory, H.fixture("softdot_regression"); atol = 1.0e-9) === :agree
end

@testitem "engine:fixture:ar_meanfield" tags = [:engine] setup = [EngineHarness] begin
    # A univariate AR(1) of three steps under mean-field, each step observed through a narrow
    # normal, with θ and γ learned from the chain.
    using ExponentialFamily, Distributions, StandardMessagePassingRules, AutoregressiveMessagePassingRules, MessagePassingRulesTestUtils
    H = EngineHarness
    meanfield(interfaces) = Tuple((first(i),) for i in interfaces)
    node!(graph, fform, interfaces; kwargs...) = H.node!(graph, fform, interfaces; factorisation = meanfield(interfaces), kwargs...)

    graph = H.Graph()
    θ, γ, x0 = H.random!(graph), H.random!(graph), H.random!(graph)
    node!(graph, NormalMeanVariance, [(:out, θ), (:μ, H.constant!(graph, 0.5)), (:v, H.constant!(graph, 1.0))])
    node!(graph, GammaShapeRate, [(:out, γ), (:α, H.constant!(graph, 2.0)), (:β, H.constant!(graph, 1.0))])
    node!(graph, NormalMeanVariance, [(:out, x0), (:μ, H.constant!(graph, 0.0)), (:v, H.constant!(graph, 1.0))])
    x, y = [H.random!(graph) for _ in 1:3], [H.data!(graph) for _ in 1:3]
    for i in 1:3
        node!(graph, AR, [(:y, x[i]), (:x, i == 1 ? x0 : x[i - 1]), (:θ, θ), (:γ, γ)]; algorithm = ARVMP(Univariate, 1, ARsafe()))
        node!(graph, NormalMeanVariance, [(:out, y[i]), (:μ, x[i]), (:v, H.constant!(graph, 0.1))])
    end

    trajectory = H.run(
        # As for softdot_regression, the posterior subscribed last updates first: v6 updated γ
        # last, from the new θ and x; with γ subscribed first, either order of θ and x reproduces it.
        graph; id = "ar_meanfield", data = y .=> [0.8, 0.5, 0.3], iterations = 5, posteriors = [:γ => γ, :θ => θ, :x => x],
        initial_marginals = [θ => NormalMeanVariance(0.5, 1.0), γ => GammaShapeRate(2.0, 1.0), x0 => NormalMeanVariance(0.0, 1.0), (x .=> Ref(NormalMeanVariance(0.0, 1.0)))...],
    )
    @test compare_engine_trajectory(trajectory, H.fixture("ar_meanfield"); atol = 1.0e-9) === :agree
end

@testitem "engine:fixture:ar2_structured" tags = [:engine] setup = [EngineHarness] begin
    # An AR(2) of three steps under the structured q(x0, x) q(θ) q(γ): each AR node's joint
    # q(y, x), the structured θ and γ rules and energy, and the ARsafe joint.
    using ExponentialFamily, Distributions, StandardMessagePassingRules, AutoregressiveMessagePassingRules, MessagePassingRulesTestUtils
    H = EngineHarness
    I2 = [1.0 0.0; 0.0 1.0]

    graph = H.Graph()
    θ, γ, x0 = H.random!(graph), H.random!(graph), H.random!(graph)
    H.node!(graph, MvNormalMeanCovariance, [(:out, θ), (:μ, H.constant!(graph, [0.5, 0.0])), (:Σ, H.constant!(graph, I2))])
    H.node!(graph, GammaShapeRate, [(:out, γ), (:α, H.constant!(graph, 2.0)), (:β, H.constant!(graph, 1.0))])
    H.node!(graph, MvNormalMeanCovariance, [(:out, x0), (:μ, H.constant!(graph, [0.0, 0.0])), (:Σ, H.constant!(graph, I2))])
    x, y = [H.random!(graph) for _ in 1:3], [H.data!(graph) for _ in 1:3]
    for i in 1:3
        H.node!(
            graph, AR, [(:y, x[i]), (:x, i == 1 ? x0 : x[i - 1]), (:θ, θ), (:γ, γ)];
            factorisation = ((:y, :x), (:θ,), (:γ,)), algorithm = ARVMP(Multivariate, 2, ARsafe()),
        )
        H.node!(graph, MvNormalMeanCovariance, [(:out, y[i]), (:μ, x[i]), (:Σ, H.constant!(graph, 0.1 * I2))])
    end

    trajectory = H.run(
        graph; id = "ar2_structured", data = y .=> [[0.8, 0.1], [0.5, 0.8], [0.3, 0.5]], iterations = 5, posteriors = [:γ => γ, :θ => θ, :x => x],
        initial_marginals = [θ => MvNormalMeanCovariance([0.5, 0.0], I2), γ => GammaShapeRate(2.0, 1.0)],
    )
    @test compare_engine_trajectory(trajectory, H.fixture("ar2_structured"); atol = 1.0e-9) === :agree
end

@testmodule ContinuousTransitionFixture begin
    # y[i] ~ N(x[i], 0.1 I) observed, x[i] ~ ContinuousTransition(x[i-1], a, W) with
    # A = reshape(a, 2, 2), as `ct_linear` in the recorder. A linear f, so every rule agrees with
    # v6's; only the energy was wrong, and the free energy is compared apart.
    using ExponentialFamily, Distributions, StandardMessagePassingRules, ContinuousTransitionMessagePassingRules, MessagePassingRulesTestUtils
    import ..EngineHarness as H

    const I2, I4 = [1.0 0.0; 0.0 1.0], [1.0 0.0 0.0 0.0; 0.0 1.0 0.0 0.0; 0.0 0.0 1.0 0.0; 0.0 0.0 0.0 1.0]
    const Y = [[1.0, 0.2], [0.8, 0.5], [0.5, 0.7]]

    function run(id; factorisation, posteriors, initial_marginals, free_energy)
        graph = H.Graph()
        a, W, x0 = H.random!(graph), H.random!(graph), H.random!(graph)
        H.node!(graph, MvNormalMeanCovariance, [(:out, a), (:μ, H.constant!(graph, [1.0, 0.0, 0.0, 1.0])), (:Σ, H.constant!(graph, I4))])
        H.node!(graph, Wishart, [(:out, W), (:ν, H.constant!(graph, 4.0)), (:S, H.constant!(graph, I2))])
        H.node!(graph, MvNormalMeanCovariance, [(:out, x0), (:μ, H.constant!(graph, [0.0, 0.0])), (:Σ, H.constant!(graph, I2))])
        x, y = [H.random!(graph) for _ in 1:3], [H.data!(graph) for _ in 1:3]
        for i in 1:3
            H.node!(
                graph, ContinuousTransition, [(:y, x[i]), (:x, i == 1 ? x0 : x[i - 1]), (:a, a), (:W, W)];
                factorisation, algorithm = CTVMP(a -> reshape(a, 2, 2)),
            )
            H.node!(graph, MvNormalMeanCovariance, [(:out, y[i]), (:μ, x[i]), (:Σ, H.constant!(graph, 0.1 * I2))])
        end
        variables = (; a, W, x0, x)
        return H.run(
            graph; id, data = y .=> Y, iterations = 5, posteriors = map(name -> name => variables[name], posteriors),
            initial_marginals = initial_marginals(variables), free_energy,
        )
    end

    # Every posterior and every rule call must agree with v6's. The free energy is set aside: v6's
    # energy was wrong, and the corrected one is pinned in the package, against the closed form and
    # a Monte Carlo estimate.
    function without_free_energy(trajectory)
        return EngineTrajectory(trajectory.id; description = trajectory.description, free_energy = Float64[], posteriors = trajectory.posteriors, trace = trajectory.trace)
    end
end

@testitem "engine:fixture:ct_structured" tags = [:engine] setup = [EngineHarness, ContinuousTransitionFixture] begin
    using ExponentialFamily, Distributions, MessagePassingRulesTestUtils
    F, H = ContinuousTransitionFixture, EngineHarness
    initial(v) = [v.a => MvNormalMeanCovariance([1.0, 0.0, 0.0, 1.0], F.I4), v.W => Wishart(4, F.I2)]
    # The free energy is subscribed to, as in the recording: it is what reads q(x0), whose message
    # no rule needs under q(x0, x). As in the other fixtures, the posterior subscribed last updates
    # first; v6 updated `a` last.
    trajectory = F.run("ct_structured"; factorisation = ((:y, :x), (:a,), (:W,)), posteriors = [:a, :W, :x], initial_marginals = initial, free_energy = true)
    @test compare_engine_trajectory(F.without_free_energy(trajectory), F.without_free_energy(H.fixture("ct_structured")); atol = 1.0e-9) === :agree
    @test length(trajectory.free_energy) == 5 && all(isfinite, trajectory.free_energy)
end

@testitem "engine:fixture:ct_meanfield" tags = [:engine] setup = [EngineHarness, ContinuousTransitionFixture] begin
    using ExponentialFamily, Distributions, MessagePassingRulesTestUtils
    F, H = ContinuousTransitionFixture, EngineHarness
    initial(v) = [
        v.a => MvNormalMeanCovariance([1.0, 0.0, 0.0, 1.0], F.I4), v.W => Wishart(4, F.I2), v.x0 => MvNormalMeanCovariance([0.0, 0.0], F.I2),
        (v.x .=> Ref(MvNormalMeanCovariance([0.0, 0.0], F.I2)))...,
    ]
    trajectory = F.run("ct_meanfield"; factorisation = ((:y,), (:x,), (:a,), (:W,)), posteriors = [:a, :W, :x], initial_marginals = initial, free_energy = true)
    @test compare_engine_trajectory(F.without_free_energy(trajectory), F.without_free_energy(H.fixture("ct_meanfield")); atol = 1.0e-9) === :agree
    @test length(trajectory.free_energy) == 5 && all(isfinite, trajectory.free_energy)
end

@testitem "engine:fixture:binomial_regression" tags = [:engine] setup = [EngineHarness] begin
    # β ~ N(0, I), y[i] ~ BinomialPolya(X[i], n[i], β): the rule towards β reads the message on its
    # own edge, which the model initialises, as `μ(β)` does in RxInfer.
    using ExponentialFamily, StandardMessagePassingRules, PolyaMessagePassingRules, MessagePassingRulesTestUtils
    H = EngineHarness
    prior = MvNormalWeightedMeanPrecision([0.0, 0.0], [1.0 0.0; 0.0 1.0])

    graph = H.Graph()
    β = H.random!(graph)
    H.node!(graph, MvNormalWeightedMeanPrecision, [(:out, β), (:ξ, H.constant!(graph, [0.0, 0.0])), (:Λ, H.constant!(graph, [1.0 0.0; 0.0 1.0]))])
    ys, Xs, ns = ([H.data!(graph) for _ in 1:3] for _ in 1:3)
    for i in 1:3
        H.node!(graph, BinomialPolya, [(:y, ys[i]), (:x, Xs[i]), (:n, ns[i]), (:β, β)])
    end

    data = [(ys .=> [3.0, 1.0, 4.0])..., (Xs .=> [[1.0, 0.5], [1.0, -0.3], [1.0, 1.2]])..., (ns .=> [5.0, 4.0, 5.0])...]
    trajectory = H.run(graph; id = "binomial_regression", data, iterations = 5, posteriors = [:β => β], initial_messages = [β => prior])
    # Every posterior and rule call agrees with v6; v6's energy took softplus at the mean of xᵀβ,
    # so the free energy is compared apart: the corrected one is higher, by Jensen's inequality.
    without_free_energy(t) = EngineTrajectory(t.id; description = t.description, free_energy = Float64[], posteriors = t.posteriors, trace = t.trace)
    v6 = H.fixture("binomial_regression")
    @test compare_engine_trajectory(without_free_energy(trajectory), without_free_energy(v6); atol = 1.0e-9) === :agree
    @test length(trajectory.free_energy) == 5 && all(trajectory.free_energy .> v6.free_energy)
end

@testitem "engine:fixture:multinomial_regression" tags = [:engine] setup = [EngineHarness] begin
    # ψ ~ N(0, I), y[i] ~ MultinomialPolya(10, ψ) with observed counts, for which v6's energy was
    # right: everything agrees, the free energy included.
    using ExponentialFamily, StandardMessagePassingRules, PolyaMessagePassingRules, MessagePassingRulesTestUtils
    H = EngineHarness
    prior = MvNormalWeightedMeanPrecision([0.0, 0.0], [1.0 0.0; 0.0 1.0])

    graph = H.Graph()
    ψ = H.random!(graph)
    H.node!(graph, MvNormalWeightedMeanPrecision, [(:out, ψ), (:ξ, H.constant!(graph, [0.0, 0.0])), (:Λ, H.constant!(graph, [1.0 0.0; 0.0 1.0]))])
    ys, N = [H.data!(graph) for _ in 1:3], H.constant!(graph, 10)
    for i in 1:3
        H.node!(graph, MultinomialPolya, [(:x, ys[i]), (:N, N), (:ψ, ψ)])
    end

    data = ys .=> [[3, 2, 5], [1, 4, 5], [2, 2, 6]]
    trajectory = H.run(graph; id = "multinomial_regression", data, iterations = 5, posteriors = [:ψ => ψ], initial_messages = [ψ => prior])
    @test compare_engine_trajectory(trajectory, H.fixture("multinomial_regression"); atol = 1.0e-9) === :agree
end

@testmodule BIFMFixture begin
    # The RxInferExamples model *RTS vs BIFM Smoothing*, small, as `bifm_smoother` in the recorder:
    # z[i + 1] = A z[i] + B u[i], y[i] = C z[i + 1] + noise, through BIFM time slices.
    using ExponentialFamily, StandardMessagePassingRules, BIFMMessagePassingRules
    import ..EngineHarness as H

    const A, B, C = [0.9 0.1; 0.0 0.8], reshape([1.0, 0.5], 2, 1), [1.0 0.0]
    const Y = [[0.5], [0.8], [0.3], [-0.1]]

    function bifm(; free_energy = false)
        graph = H.Graph()
        z_prior = H.random!(graph)
        z = [H.random!(graph) for _ in 1:5]
        u, yt, y = [H.random!(graph) for _ in 1:4], [H.random!(graph) for _ in 1:4], [H.data!(graph) for _ in 1:4]
        H.node!(graph, MvNormalMeanPrecision, [(:out, z_prior), (:μ, H.constant!(graph, [0.0, 0.0])), (:Λ, H.constant!(graph, [1.0e-5 0.0; 0.0 1.0e-5]))])
        H.node!(graph, BIFMHelper, [(:out, z[1]), (:in, z_prior)]; factorisation = ((:out,), (:in,)))
        for i in 1:4
            H.node!(graph, MvNormalMeanPrecision, [(:out, u[i]), (:μ, H.constant!(graph, [0.0])), (:Λ, H.constant!(graph, [1.0;;]))])
            H.node!(graph, BIFM, [(:out, yt[i]), (:in, u[i]), (:zprev, z[i]), (:znext, z[i + 1])]; algorithm = BIFMSmoother(A, B, C))
            H.node!(graph, MvNormalMeanPrecision, [(:out, y[i]), (:μ, yt[i]), (:Λ, H.constant!(graph, [10.0;;]))])
        end
        H.node!(graph, MvNormalMeanPrecision, [(:out, z[5]), (:μ, H.constant!(graph, [0.0, 0.0])), (:Λ, H.constant!(graph, [0.0 0.0; 0.0 0.0]))])
        # `yt`'s posterior too: it is what reads the messages towards `out`, which v6 computed
        # whether or not anything asked for them.
        return H.run(graph; id = "bifm_smoother", data = y .=> Y, iterations = 1, posteriors = [:z => z, :u => u, :yt => yt], free_energy)
    end
end

@testitem "engine:fixture:bifm_smoother" tags = [:engine] setup = [EngineHarness, BIFMFixture] begin
    using MessagePassingRulesTestUtils
    # v6's rule towards `in` read what its rule towards `znext` had cached, so v6 ran `znext` first.
    # The port's rules read their own messages and do not depend on each other; the engine runs a
    # slice's `in` before its `znext`, and the order within an iteration is declared free.
    @test compare_engine_trajectory(BIFMFixture.bifm(), EngineHarness.fixture("bifm_smoother"); atol = 1.0e-9, trace_order = :within_iteration) === :agree
end

@testitem "engine:bifm is the RTS smoother" tags = [:engine] setup = [EngineHarness, BIFMFixture] begin
    # The same linear-Gaussian model through Standard's nodes, z[i] = A z[i - 1] + B u[i] and
    # y[i] ~ N(C z[i], 1/10): belief propagation on a chain is exact, and BIFM must agree, v6 aside.
    using ExponentialFamily, BayesBase, StandardMessagePassingRules
    F, H = BIFMFixture, EngineHarness

    graph = H.Graph()
    z_prev = H.random!(graph)
    H.node!(graph, MvNormalMeanPrecision, [(:out, z_prev), (:μ, H.constant!(graph, [0.0, 0.0])), (:Λ, H.constant!(graph, [1.0e-5 0.0; 0.0 1.0e-5]))])
    z, u, y = [H.random!(graph) for _ in 1:4], [H.random!(graph) for _ in 1:4], [H.data!(graph) for _ in 1:4]
    for i in 1:4
        H.node!(graph, MvNormalMeanPrecision, [(:out, u[i]), (:μ, H.constant!(graph, [0.0])), (:Λ, H.constant!(graph, [1.0;;]))])
        Az, Bu, Cz = H.random!(graph), H.random!(graph), H.random!(graph)
        H.node!(graph, *, [(:out, Az), (:A, H.constant!(graph, F.A)), (:in, i == 1 ? z_prev : z[i - 1])])
        H.node!(graph, *, [(:out, Bu), (:A, H.constant!(graph, F.B)), (:in, u[i])])
        H.node!(graph, +, [(:out, z[i]), (:in1, Az), (:in2, Bu)])
        H.node!(graph, *, [(:out, Cz), (:A, H.constant!(graph, F.C)), (:in, z[i])])
        H.node!(graph, MvNormalMeanPrecision, [(:out, y[i]), (:μ, Cz), (:Λ, H.constant!(graph, [10.0;;]))])
    end
    rts = H.run(graph; id = "rts", data = y .=> F.Y, iterations = 1, posteriors = [:z => z, :u => u], free_energy = false)
    bifm = F.bifm()

    # BIFM's z[1] is the chain's start, so its z[2:5] are the RTS smoother's z[1:4].
    unwrap(q) = q isa BayesBase.TerminalProdArgument ? q.argument : q
    for (q_bifm, q_rts) in zip(bifm.posteriors["z"][2:5], rts.posteriors["z"])
        @test mean(unwrap(q_bifm)) ≈ mean(q_rts) atol = 1.0e-8
        @test cov(unwrap(q_bifm)) ≈ cov(q_rts) atol = 1.0e-8
    end
    for (q_bifm, q_rts) in zip(bifm.posteriors["u"], rts.posteriors["u"])
        @test mean(unwrap(q_bifm)) ≈ mean(q_rts) atol = 1.0e-8
        @test cov(unwrap(q_bifm)) ≈ cov(q_rts) atol = 1.0e-8
    end
end

@testitem "engine:bifm's free energy is an error" tags = [:engine] setup = [EngineHarness, BIFMFixture] begin
    # v6's failed with an `Inf`; the port says what is not supported.
    using BIFMMessagePassingRules
    @test_throws BIFMMessagePassingRules.BIFMFreeEnergyError BIFMFixture.bifm(free_energy = true)
end

@testmodule FlowFixture begin
    # The RxInferExamples *Invertible Neural Network Tutorial*'s first model, as `flow_meanfield` in
    # the recorder: x[k] ~ N(z_μ, z_Λ⁻¹), y_lat[k] = f(x[k]), y[k] ~ N(y_lat[k], 0.01 I).
    using ExponentialFamily, Distributions, StandardMessagePassingRules, FlowMessagePassingRules
    import MessagePassingRulesApproximations as A
    import ..EngineHarness as H

    const MODEL = compile(
        FlowModel((InputLayer(2), AdditiveCouplingLayer(PlanarFlow(); permute = false), PermutationLayer(PermutationMatrix([2, 1])), AdditiveCouplingLayer(PlanarFlow(); permute = false))),
        [0.3, -0.2, 0.5, 0.1, 0.4, -0.3],
    )
    const Y = [[1.2, 0.3], [0.8, -0.4], [1.5, 0.9], [0.2, 0.1]]
    const I2 = [1.0 0.0; 0.0 1.0]

    function run(id, method; posteriors = [:z_μ, :z_Λ, :x], iterations = 5)
        graph = H.Graph()
        z_μ, z_Λ = H.random!(graph), H.random!(graph)
        H.node!(graph, MvNormalMeanCovariance, [(:out, z_μ), (:μ, H.constant!(graph, [0.0, 0.0])), (:Σ, H.constant!(graph, 100 * I2))])
        H.node!(graph, Wishart, [(:out, z_Λ), (:ν, H.constant!(graph, 3.0)), (:S, H.constant!(graph, I2))])
        x, y_lat, y = [H.random!(graph) for _ in 1:4], [H.random!(graph) for _ in 1:4], [H.data!(graph) for _ in 1:4]
        for k in 1:4
            H.node!(graph, MvNormalMeanPrecision, [(:out, x[k]), (:μ, z_μ), (:Λ, z_Λ)]; factorisation = ((:out,), (:μ,), (:Λ,)))
            H.node!(graph, Flow, [(:out, y_lat[k]), (:in, x[k])]; algorithm = FlowApproximation(MODEL; method))
            H.node!(graph, MvNormalMeanCovariance, [(:out, y[k]), (:μ, y_lat[k]), (:Σ, H.constant!(graph, 0.01 * I2))])
        end
        variables = (; z_μ, z_Λ, x)
        return H.run(
            graph; id, data = y .=> Y, iterations, posteriors = map(name -> name => variables[name], posteriors),
            initial_marginals = [z_μ => MvNormalMeanCovariance([0.0, 0.0], 100 * I2), z_Λ => Wishart(3.0, I2)],
        )
    end
end

# Every posterior and rule call agrees with v6. The free energy agrees once the posteriors settle,
# not along the way: a single-input deterministic node's entropy term is `in`'s own marginal, as
# current as its messages, where v6 recomputed it from its marginal rule only once both of the
# node's messages had refreshed. Run for 40 iterations, both engines reach the same value, v6's
# given here as recorded in 6.5.0.
@testitem "engine:fixture:flow_meanfield" tags = [:engine] setup = [EngineHarness, FlowFixture] begin
    using MessagePassingRulesTestUtils
    import MessagePassingRulesApproximations as A
    strip(t) = EngineTrajectory(t.id; description = t.description, free_energy = Float64[], posteriors = t.posteriors, trace = t.trace)
    # Posteriors and rule calls exactly; the free energy along the way differs (see above).
    @test compare_engine_trajectory(strip(FlowFixture.run("flow_meanfield", A.Linearization())), strip(EngineHarness.fixture("flow_meanfield")); atol = 1.0e-9) === :agree
    @test last(FlowFixture.run("flow_meanfield", A.Linearization(); iterations = 40).free_energy) ≈ 14.971494010931867 atol = 1.0e-9
end

@testitem "engine:fixture:flow_meanfield_unscented" tags = [:engine] setup = [EngineHarness, FlowFixture] begin
    using MessagePassingRulesTestUtils
    import MessagePassingRulesApproximations as A
    strip(t) = EngineTrajectory(t.id; description = t.description, free_energy = Float64[], posteriors = t.posteriors, trace = t.trace)
    @test compare_engine_trajectory(strip(FlowFixture.run("flow_meanfield_unscented", A.Unscented(2))), strip(EngineHarness.fixture("flow_meanfield_unscented")); atol = 1.0e-9) === :agree
    @test last(FlowFixture.run("flow_meanfield_unscented", A.Unscented(2); iterations = 40).free_energy) ≈ 14.977626369022019 atol = 1.0e-8
end

@testitem "engine:fixture:dt_hmm" tags = [:engine] setup = [EngineHarness] begin
    # RxInferExamples' Hidden Markov Model under q(s0, s) q(A) q(B): each transition's joint
    # q(out, in), the emissions' messages from an observed `out`, the learned tensors and the energy.
    using ExponentialFamily, Distributions, StandardMessagePassingRules, DiscreteTransitionMessagePassingRules, MessagePassingRulesTestUtils
    H = EngineHarness
    I3 = [1.1 0.1 0.1; 0.1 1.1 0.1; 0.1 0.1 1.1]
    observed = [[1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0]]

    graph = H.Graph()
    A, B, s0 = H.random!(graph), H.random!(graph), H.random!(graph)
    H.node!(graph, DirichletCollection, [(:out, A), (:a, H.constant!(graph, ones(3, 3)))])
    H.node!(graph, DirichletCollection, [(:out, B), (:a, H.constant!(graph, [10.0 1.0 1.0; 1.0 10.0 1.0; 1.0 1.0 10.0]))])
    H.node!(graph, Categorical, [(:out, s0), (:p, H.constant!(graph, fill(1.0 / 3.0, 3)))])
    s, x = [H.random!(graph) for _ in observed], [H.data!(graph) for _ in observed]
    for t in eachindex(observed)
        H.node!(graph, DiscreteTransition, [(:out, s[t]), (:in, t == 1 ? s0 : s[t - 1]), (:a, A)]; factorisation = ((:out, :in), (:a,)))
        H.node!(graph, DiscreteTransition, [(:out, x[t]), (:in, s[t]), (:a, B)]; factorisation = ((:out,), (:in,), (:a,)))
    end

    trajectory = H.run(
        graph; id = "dt_hmm", data = x .=> observed, iterations = 5, posteriors = [:A => A, :B => B, :s => s],
        initial_marginals = [A => DirichletCollection(I3), B => DirichletCollection(I3)],
    )
    @test compare_engine_trajectory(trajectory, H.fixture("dt_hmm"); atol = 1.0e-9) === :agree
end

@testitem "engine:fixture:dt_partial_joint" tags = [:engine] setup = [EngineHarness] begin
    # A node with two `T`s under q(out, T1) q(in) q(T2) q(a): a joint of `out` and only the first
    # member of the group, its marginal, the messages in and out of it, and the energy.
    using ExponentialFamily, Distributions, StandardMessagePassingRules, DiscreteTransitionMessagePassingRules, MessagePassingRulesTestUtils
    H = EngineHarness

    graph = H.Graph()
    A = H.random!(graph)
    H.node!(graph, DirichletCollection, [(:out, A), (:a, H.constant!(graph, ones(2, 2, 3, 2)))])
    x, t1, t2 = H.random!(graph), H.random!(graph), H.random!(graph)
    H.node!(graph, Categorical, [(:out, x), (:p, H.constant!(graph, [0.4, 0.6]))])
    H.node!(graph, Categorical, [(:out, t1), (:p, H.constant!(graph, [0.2, 0.3, 0.5]))])
    H.node!(graph, Categorical, [(:out, t2), (:p, H.constant!(graph, [0.5, 0.5]))])
    z, y = H.random!(graph), H.data!(graph)
    H.node!(
        graph, DiscreteTransition, [(:out, z), (:in, x), (:a, A), ((:T, 1), t1), ((:T, 2), t2)];
        factorisation = ((:out, (:T, 1)), (:in,), (:a,), ((:T, 2),)),
    )
    H.node!(graph, DiscreteTransition, [(:out, y), (:in, z), (:a, H.constant!(graph, [0.9 0.2; 0.1 0.8]))])

    # The posteriors in the order RxInfer subscribed to them, its `Dict`'s: under VMP the order
    # marginals are first asked for is the schedule, and it moves the result by 1e-8.
    trajectory = H.run(
        graph; id = "dt_partial_joint", data = [y => [0.0, 1.0]], iterations = 5, posteriors = [:t2 => t2, :A => A, :z => z, :t1 => t1, :x => x],
        initial_marginals = [A => DirichletCollection(ones(2, 2, 3, 2)), x => Categorical([0.5, 0.5]), t2 => Categorical([0.5, 0.5])],
    )
    # v6 named the members of the group as interfaces of their own, `T2` for `(:T, 2)`.
    v6_named(r) = RuleCallRecord(r.iteration, r.node, replace(r.target, r"^\(:T, (\d+)\)$" => s":T\1"), r.result, r.logscale)
    trajectory = EngineTrajectory(trajectory.id; free_energy = trajectory.free_energy, posteriors = trajectory.posteriors, trace = map(v6_named, trajectory.trace))
    @test compare_engine_trajectory(trajectory, H.fixture("dt_partial_joint"); atol = 1.0e-9) === :agree
end

@testitem "engine:fixture:arithmetic_bp" tags = [:engine] setup = [EngineHarness] begin
    # Belief propagation through `+`, `-`, `*` by a constant scalar and by a constant matrix, and
    # `dot` with a constant vector: each node's messages both ways, and its joint over its inputs
    # in the free energy.
    using ExponentialFamily, LinearAlgebra, StandardMessagePassingRules, MessagePassingRulesTestUtils
    H = EngineHarness

    graph = H.Graph()
    x = H.random!(graph)
    x_prior = [(:out, x), (:μ, H.constant!(graph, 0.0)), (:v, H.constant!(graph, 1.0))]
    z = H.random!(graph)
    z_prior = [(:out, z), (:μ, H.constant!(graph, 1.0)), (:v, H.constant!(graph, 2.0))]
    s, w = H.random!(graph), H.random!(graph)
    w_prior = [(:out, w), (:μ, H.constant!(graph, 0.5)), (:v, H.constant!(graph, 1.0))]
    d, m = H.random!(graph), H.random!(graph)
    two, y1 = H.constant!(graph, 2.0), H.data!(graph)
    v = H.random!(graph)
    v_prior = [(:out, v), (:μ, H.constant!(graph, [0.0, 1.0])), (:Σ, H.constant!(graph, [1.0 0.0; 0.0 2.0]))]
    Av, A = H.random!(graph), H.constant!(graph, [1.0 0.5; 0.0 1.0])
    q, c = H.random!(graph), H.constant!(graph, [1.0, 2.0])
    y2 = H.data!(graph)

    H.node!(graph, NormalMeanVariance, x_prior)
    H.node!(graph, NormalMeanVariance, z_prior)
    H.node!(graph, +, [(:out, s), (:in1, x), (:in2, z)])
    H.node!(graph, NormalMeanVariance, w_prior)
    H.node!(graph, -, [(:out, d), (:in1, s), (:in2, w)])
    H.node!(graph, *, [(:out, m), (:A, two), (:in, d)])
    H.node!(graph, NormalMeanVariance, [(:out, y1), (:μ, m), (:v, H.constant!(graph, 0.5))])
    H.node!(graph, MvNormalMeanCovariance, v_prior)
    H.node!(graph, *, [(:out, Av), (:A, A), (:in, v)])
    H.node!(graph, dot, [(:out, q), (:in1, c), (:in2, Av)])
    H.node!(graph, NormalMeanVariance, [(:out, y2), (:μ, q), (:v, H.constant!(graph, 0.5))])

    # The posteriors in the order RxInfer subscribed to them, its `Dict`'s.
    variables = (; x, z, s, w, d, m, v, Av, q)
    trajectory = H.run(
        graph; id = "arithmetic_bp", data = [y1 => 1.5, y2 => 2.0], iterations = 2,
        posteriors = [name => variables[name] for name in (:w, :m, :d, :s, :v, :Av, :z, :q, :x)],
    )
    @test compare_engine_trajectory(trajectory, H.fixture("arithmetic_bp"); atol = 1.0e-9) === :agree
end

@testitem "engine:fixture:bernoulli_priors" tags = [:engine] setup = [EngineHarness] begin
    # Beta and Uniform priors of Bernoulli observations: each prior's message, its product with
    # the likelihoods' Beta messages, and the energies.
    using ExponentialFamily, Distributions, StandardMessagePassingRules, MessagePassingRulesTestUtils
    H = EngineHarness

    graph = H.Graph()
    p = H.random!(graph)
    p_prior = [(:out, p), (:a, H.constant!(graph, 2.0)), (:b, H.constant!(graph, 3.0))]
    r = H.random!(graph)
    r_prior = [(:out, r), (:a, H.constant!(graph, 0.0)), (:b, H.constant!(graph, 1.0))]
    y, u = [H.data!(graph) for _ in 1:3], [H.data!(graph) for _ in 1:3]
    H.node!(graph, Beta, p_prior)
    H.node!(graph, Uniform, r_prior)
    for i in 1:3
        H.node!(graph, Bernoulli, [(:out, y[i]), (:p, p)])
        H.node!(graph, Bernoulli, [(:out, u[i]), (:p, r)])
    end

    trajectory = H.run(
        graph; id = "bernoulli_priors", data = [y => [1.0, 0.0, 1.0], u => [0.0, 0.0, 1.0]], iterations = 2,
        posteriors = [:p => p, :r => r],
    )
    @test compare_engine_trajectory(trajectory, H.fixture("bernoulli_priors"); atol = 1.0e-9) === :agree
end

@testitem "engine:fixture:poisson_gamma" tags = [:engine] setup = [EngineHarness] begin
    # Poisson counts under a Gamma (shape, scale) prior: the messages towards the rate, their
    # product with the prior's, and both nodes' energies.
    using ExponentialFamily, Distributions, StandardMessagePassingRules, MessagePassingRulesTestUtils
    H = EngineHarness

    graph = H.Graph()
    l = H.random!(graph)
    l_prior = [(:out, l), (:α, H.constant!(graph, 2.0)), (:θ, H.constant!(graph, 1.5))]
    y = [H.data!(graph) for _ in 1:3]
    H.node!(graph, Gamma, l_prior)
    for i in 1:3
        H.node!(graph, Poisson, [(:out, y[i]), (:l, l)])
    end

    trajectory = H.run(graph; id = "poisson_gamma", data = [y => [3, 1, 4]], iterations = 2, posteriors = [:l => l])
    @test compare_engine_trajectory(trajectory, H.fixture("poisson_gamma"); atol = 1.0e-9) === :agree
end

@testitem "engine:fixture:gamma_inverse_variance" tags = [:engine] setup = [EngineHarness] begin
    # An inverse-gamma prior on a known-mean normal's variance: the prior's message and the
    # likelihoods' messages towards `v` agree with v6, and so does q(v). v6's GammaInverse energy
    # took θ/E[x] for E[θ/x] (ReactiveMP.jl#672), wrong for this q(v), so the free energy is
    # compared apart: the port's is pinned here, against the closed form.
    using ExponentialFamily, Distributions, SpecialFunctions, StandardMessagePassingRules, MessagePassingRulesTestUtils
    H = EngineHarness

    graph = H.Graph()
    v = H.random!(graph)
    v_prior = [(:out, v), (:α, H.constant!(graph, 3.0)), (:θ, H.constant!(graph, 2.0))]
    y = [H.data!(graph) for _ in 1:3]
    H.node!(graph, GammaInverse, v_prior)
    for i in 1:3
        H.node!(graph, NormalMeanVariance, [(:out, y[i]), (:μ, H.constant!(graph, 1.0)), (:v, v)])
    end

    trajectory = H.run(graph; id = "gamma_inverse_variance", data = [y => [1.2, 0.3, 2.1]], iterations = 2, posteriors = [:v => v])
    without_free_energy(t) = EngineTrajectory(t.id; description = t.description, free_energy = Float64[], posteriors = t.posteriors, trace = t.trace)
    v6 = H.fixture("gamma_inverse_variance")
    @test compare_engine_trajectory(without_free_energy(trajectory), without_free_energy(v6); atol = 1.0e-9) === :agree
    # Exact: q(v) is the exact posterior, so the free energy is -log p(y), for S = Σ(y - 1)² = 1.74
    # 3 log 2 - log Γ(3) + log Γ(4.5) - 4.5 log(2 + S/2) - 1.5 log 2π, and v6's was not.
    S = sum(abs2, [1.2, 0.3, 2.1] .- 1.0)
    evidence = 3 * log(2.0) - loggamma(3.0) + loggamma(4.5) - 4.5 * log(2.0 + S / 2) - 1.5 * log(2π)
    @test trajectory.free_energy ≈ [-evidence, -evidence] atol = 1.0e-9
    @test trajectory.free_energy ≈ [3.6611888016235694, 3.6611888016235694] atol = 1.0e-9
    @test !(trajectory.free_energy ≈ v6.free_energy)
end

@testitem "engine:fixture:matrix_normal_covariances" tags = [:engine] setup = [EngineHarness] begin
    # MatrixNormal's rules towards its row and column covariances under mean-field, each reading
    # the other's inverse-Wishart marginal, and the InverseWishart priors.
    using ExponentialFamily, Distributions, StandardMessagePassingRules, MessagePassingRulesTestUtils
    H = EngineHarness
    meanfield!(graph, fform, interfaces) = H.node!(graph, fform, interfaces; factorisation = H.meanfield_factorisation(interfaces))
    I2 = [1.0 0.0; 0.0 1.0]

    graph = H.Graph()
    U = H.random!(graph)
    U_prior = [(:out, U), (:ν, H.constant!(graph, 5.0)), (:S, H.constant!(graph, I2))]
    V = H.random!(graph)
    V_prior = [(:out, V), (:ν, H.constant!(graph, 4.0)), (:S, H.constant!(graph, 2 * I2))]
    y = [H.data!(graph) for _ in 1:3]
    meanfield!(graph, InverseWishart, U_prior)
    meanfield!(graph, InverseWishart, V_prior)
    for i in 1:3
        meanfield!(graph, MatrixNormal, [(:out, y[i]), (:M, H.constant!(graph, [0.5 0.0; 0.0 0.5])), (:U, U), (:V, V)])
    end

    Y = [[1.0 0.5; 0.2 1.1], [0.8 0.3; -0.1 0.9], [0.2 -0.4; 0.6 0.3]]
    trajectory = H.run(
        graph; id = "matrix_normal_covariances", data = [y => Y], iterations = 5, posteriors = [:U => U, :V => V],
        initial_marginals = [U => InverseWishart(5.0, I2), V => InverseWishart(4.0, 2 * I2)],
    )
    @test compare_engine_trajectory(trajectory, H.fixture("matrix_normal_covariances"); atol = 1.0e-9) === :agree
end

@testitem "engine:fixture:mv_scale_precision" tags = [:engine] setup = [EngineHarness] begin
    # MvNormalMeanScalePrecision under mean-field, its mean and scale learned.
    using ExponentialFamily, StandardMessagePassingRules, MessagePassingRulesTestUtils
    H = EngineHarness
    meanfield!(graph, fform, interfaces) = H.node!(graph, fform, interfaces; factorisation = H.meanfield_factorisation(interfaces))

    graph = H.Graph()
    μ = H.random!(graph)
    μ_prior = [(:out, μ), (:μ, H.constant!(graph, [0.0, 0.0])), (:Σ, H.constant!(graph, [10.0 0.0; 0.0 10.0]))]
    γ = H.random!(graph)
    γ_prior = [(:out, γ), (:α, H.constant!(graph, 2.0)), (:β, H.constant!(graph, 1.0))]
    y = [H.data!(graph) for _ in 1:3]
    meanfield!(graph, MvNormalMeanCovariance, μ_prior)
    meanfield!(graph, GammaShapeRate, γ_prior)
    for i in 1:3
        meanfield!(graph, MvNormalMeanScalePrecision, [(:out, y[i]), (:μ, μ), (:γ, γ)])
    end

    trajectory = H.run(
        graph; id = "mv_scale_precision", data = [y => [[1.0, 0.5], [0.3, 1.2], [0.8, 0.9]]], iterations = 5,
        posteriors = [:γ => γ, :μ => μ], initial_marginals = [γ => GammaShapeRate(2.0, 1.0)],
    )
    @test compare_engine_trajectory(trajectory, H.fixture("mv_scale_precision"); atol = 1.0e-9) === :agree
end

@testitem "engine:fixture:mv_scale_matrix_precision" tags = [:engine] setup = [EngineHarness] begin
    # MvNormalMeanScaleMatrixPrecision under mean-field, its mean, scale and matrix learned.
    using ExponentialFamily, Distributions, StandardMessagePassingRules, MessagePassingRulesTestUtils
    H = EngineHarness
    meanfield!(graph, fform, interfaces) = H.node!(graph, fform, interfaces; factorisation = H.meanfield_factorisation(interfaces))
    I2 = [1.0 0.0; 0.0 1.0]

    graph = H.Graph()
    μ = H.random!(graph)
    μ_prior = [(:out, μ), (:μ, H.constant!(graph, [0.0, 0.0])), (:Σ, H.constant!(graph, 10 * I2))]
    γ = H.random!(graph)
    γ_prior = [(:out, γ), (:α, H.constant!(graph, 2.0)), (:β, H.constant!(graph, 1.0))]
    G = H.random!(graph)
    G_prior = [(:out, G), (:ν, H.constant!(graph, 3.0)), (:S, H.constant!(graph, I2))]
    y = [H.data!(graph) for _ in 1:3]
    meanfield!(graph, MvNormalMeanCovariance, μ_prior)
    meanfield!(graph, GammaShapeRate, γ_prior)
    meanfield!(graph, Wishart, G_prior)
    for i in 1:3
        meanfield!(graph, MvNormalMeanScaleMatrixPrecision, [(:out, y[i]), (:μ, μ), (:γ, γ), (:G, G)])
    end

    trajectory = H.run(
        graph; id = "mv_scale_matrix_precision", data = [y => [[1.0, 0.5], [0.3, 1.2], [0.8, 0.9]]], iterations = 5,
        posteriors = [:γ => γ, :μ => μ, :G => G], initial_marginals = [γ => GammaShapeRate(2.0, 1.0), G => Wishart(3.0, I2)],
    )
    @test compare_engine_trajectory(trajectory, H.fixture("mv_scale_matrix_precision"); atol = 1.0e-9) === :agree
end

@testitem "engine:fixture:conjugate_ar2_structured" tags = [:engine] setup = [EngineHarness] begin
    # ar2_structured with (θ, γ) joint on ConjugateAR's `w`, under q(x0, x) q(w): each node's
    # joint q(y, x), the rules towards `y`, `x` and `w`, the MvNormalGamma prior and the energies.
    using ExponentialFamily, Distributions, StandardMessagePassingRules, AutoregressiveMessagePassingRules, MessagePassingRulesTestUtils
    H = EngineHarness
    I2 = [1.0 0.0; 0.0 1.0]

    graph = H.Graph()
    w = H.random!(graph)
    w_prior = [(:out, w), (:μ, H.constant!(graph, [0.5, 0.0])), (:Λ, H.constant!(graph, I2)), (:α, H.constant!(graph, 2.0)), (:β, H.constant!(graph, 1.0))]
    x0 = H.random!(graph)
    x0_prior = [(:out, x0), (:μ, H.constant!(graph, [0.0, 0.0])), (:Σ, H.constant!(graph, I2))]
    H.node!(graph, MvNormalGamma, w_prior)
    H.node!(graph, MvNormalMeanCovariance, x0_prior)
    x, y = [H.random!(graph) for _ in 1:3], [H.data!(graph) for _ in 1:3]
    for i in 1:3
        H.node!(
            graph, ConjugateAR, [(:y, x[i]), (:x, i == 1 ? x0 : x[i - 1]), (:w, w)];
            factorisation = ((:y, :x), (:w,)), algorithm = ARVMP(Multivariate, 2, ARsafe()),
        )
        H.node!(graph, MvNormalMeanCovariance, [(:out, y[i]), (:μ, x[i]), (:Σ, H.constant!(graph, 0.1 * I2))])
    end

    trajectory = H.run(
        graph; id = "conjugate_ar2_structured", data = y .=> [[0.8, 0.1], [0.5, 0.8], [0.3, 0.5]], iterations = 5, posteriors = [:w => w, :x => x],
        initial_marginals = [w => MvNormalGamma([0.5, 0.0], I2, 2.0, 1.0)],
    )
    @test compare_engine_trajectory(trajectory, H.fixture("conjugate_ar2_structured"); atol = 1.0e-9) === :agree
end

@testitem "engine:fixture:gamma_mixture" tags = [:engine] setup = [EngineHarness] begin
    # A two-component GammaMixture under mean-field, the shapes known and the rates learned: the
    # rules towards the switch and each rate, and the energy.
    using ExponentialFamily, Distributions, StandardMessagePassingRules, MessagePassingRulesTestUtils
    H = EngineHarness
    meanfield!(graph, fform, interfaces; kwargs...) = H.node!(graph, fform, interfaces; factorisation = H.meanfield_factorisation(interfaces), kwargs...)
    Y = [0.3, 0.5, 2.1, 1.8, 0.4, 2.5]

    graph = H.Graph()
    π = H.random!(graph)
    π_prior = [(:out, π), (:a, H.constant!(graph, [1.0, 1.0]))]
    b, b_priors = [], []
    for (α, β) in ((2.0, 4.0), (2.0, 1.0))
        push!(b, H.random!(graph))
        push!(b_priors, [(:out, b[end]), (:α, H.constant!(graph, α)), (:β, H.constant!(graph, β))])
    end
    z, y = [], []
    for _ in Y
        push!(z, H.random!(graph))
        push!(y, H.data!(graph))
    end
    a1, a2 = H.data!(graph), H.data!(graph)
    meanfield!(graph, Dirichlet, π_prior)
    foreach(prior -> meanfield!(graph, GammaShapeRate, prior), b_priors)
    for i in eachindex(Y)
        meanfield!(graph, Categorical, [(:out, z[i]), (:p, π)])
        meanfield!(graph, GammaMixture, [(:out, y[i]), (:switch, z[i]), ((:a, 1), a1), ((:a, 2), a2), ((:b, 1), b[1]), ((:b, 2), b[2])])
    end

    trajectory = H.run(
        graph; id = "gamma_mixture", data = [y => Y, a1 => 2.0, a2 => 5.0], iterations = 5,
        posteriors = [:b => b, :π => π, :z => z],
        initial_marginals = [π => Dirichlet([1.0, 1.0]), b[1] => GammaShapeRate(1.0, 1.0), b[2] => GammaShapeRate(1.0, 1.0)],
    )
    # As in normal_mixture, v6 subscribed to a group's members last first, and the engine in
    # declaration order: v6 updated π, b[2] and b[1] after z, the engine b[1], π and b[2]. Each
    # reads q(z) and none reads another, so this reorders calls inside an iteration and changes no
    # value (`DISCUSSION.md` §3.24).
    @test compare_engine_trajectory(trajectory, H.fixture("gamma_mixture"); atol = 1.0e-9, trace_order = :within_iteration) === :agree
end

@testmodule UninformativeRecords begin
    # v6 recorded its Uninformative message by name, not being a distribution; so is the port's.
    using MessagePassingRulesTestUtils: EngineTrajectory, RuleCallRecord
    using StandardMessagePassingRules: Uninformative
    named(r) = r.result isa Uninformative ? RuleCallRecord(r.iteration, r.node, r.target, Dict{String, Any}("type" => "Uninformative"), r.logscale) : r
    by_name(t) = EngineTrajectory(t.id; description = t.description, free_energy = t.free_energy, posteriors = t.posteriors, trace = map(named, t.trace))
end

@testitem "engine:fixture:half_normal_uninformative" tags = [:engine] setup = [EngineHarness, UninformativeRecords] begin
    # HalfNormal's message, from a variance given as data, closed by an Uninformative factor; an
    # Uninformative prior of an observed normal's mean; both nodes' energies.
    using ExponentialFamily, Distributions, StandardMessagePassingRules, MessagePassingRulesTestUtils
    H = EngineHarness

    graph = H.Graph()
    v0, h = H.data!(graph), H.random!(graph)
    u, y = H.random!(graph), H.data!(graph)
    H.node!(graph, HalfNormal, [(:out, h), (:v, v0)])
    H.node!(graph, Uninformative, [(:out, h)])
    H.node!(graph, Uninformative, [(:out, u)])
    H.node!(graph, NormalMeanVariance, [(:out, y), (:μ, u), (:v, H.constant!(graph, 1.0))])

    trajectory = H.run(graph; id = "half_normal_uninformative", data = [v0 => 2.0, y => 1.5], iterations = 2, posteriors = [:h => h, :u => u])
    @test compare_engine_trajectory(UninformativeRecords.by_name(trajectory), H.fixture("half_normal_uninformative"); atol = 1.0e-9) === :agree
end

@testitem "engine:fixture:normal_wishart_priors" tags = [:engine] setup = [EngineHarness, UninformativeRecords] begin
    # MvNormalWishart's and MatrixNormalWishart's messages, from a parameter given as data, each
    # closed by an Uninformative factor. No free energy: v6 could compute neither node's.
    using ExponentialFamily, Distributions, StandardMessagePassingRules, MessagePassingRulesTestUtils
    H = EngineHarness
    I2 = [1.0 0.0; 0.0 1.0]

    graph = H.Graph()
    μ0, nw = H.data!(graph), H.random!(graph)
    M0, mw = H.data!(graph), H.random!(graph)
    nw_prior = [(:out, nw), (:μ, μ0), (:W, H.constant!(graph, I2)), (:λ, H.constant!(graph, 2.0)), (:ν, H.constant!(graph, 3.0))]
    mw_prior = [(:out, mw), (:M, M0), (:U, H.constant!(graph, I2)), (:V, H.constant!(graph, I2)), (:ν, H.constant!(graph, 3.0))]
    H.node!(graph, MvNormalWishart, nw_prior; factorisation = H.meanfield_factorisation(nw_prior))
    H.node!(graph, Uninformative, [(:out, nw)])
    H.node!(graph, MatrixNormalWishart, mw_prior; factorisation = H.meanfield_factorisation(mw_prior))
    H.node!(graph, Uninformative, [(:out, mw)])

    trajectory = H.run(
        graph; id = "normal_wishart_priors", data = [μ0 => [0.5, 1.0], M0 => [1.0 0.5; 0.0 1.0]], iterations = 2,
        posteriors = [:nw => nw, :mw => mw], free_energy = false,
    )
    @test compare_engine_trajectory(UninformativeRecords.by_name(trajectory), H.fixture("normal_wishart_priors"); atol = 1.0e-9) === :agree
end
