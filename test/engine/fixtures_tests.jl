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
