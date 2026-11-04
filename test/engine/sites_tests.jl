# The site stage of a random variable, cavity propagation: with a site policy each edge's inbound
# message is replaced by its site, computed against the variable's outbound message on the edge.
# A toy policy moment-matches a likelihood on a grid: expectation propagation.

@testmodule SitePolicies begin
    using ReactiveMP, BayesBase, ExponentialFamily, Distributions, MessagePassingRulesBase

    # Expectation propagation on a grid, for messages that are univariate log-densities; any other
    # message is its own site. `projections` counts the sites computed.
    mutable struct GridEP
        tol::Float64
        projections::Int
    end
    GridEP() = GridEP(1.0e-11, 0)

    const GRID = range(-12.0, 12.0; length = 40_001)

    function ReactiveMP.compute_site(policy::GridEP, variable, index, message, cavity, previous)
        message isa ContinuousUnivariateLogPdf || return (message, nothing)
        ismissing(cavity) && return (missing, previous)
        policy.projections += 1
        ξ, w = weightedmean_precision(cavity)
        logt = [logpdf(message, x) + ξ * x - w * x^2 / 2 for x in GRID]
        p = exp.(logt .- maximum(logt))
        p ./= sum(p)
        m = sum(p .* GRID)
        v = sum(p .* (GRID .- m) .^ 2)
        site = NormalWeightedMeanPrecision(m / v - ξ, 1 / v - w)
        if previous isa NormalWeightedMeanPrecision
            Δ = max(abs(site.xi - previous.xi), abs(site.w - previous.w))
            Δ <= policy.tol * (1 + abs(site.xi) + abs(site.w)) && return (nothing, previous)
        end
        return (site, site)
    end

    # `y ~ Logit(x)`: a node with no rules, whose likelihood of x is log σ(±x); a toy fallback sends
    # it, and its free-energy term, EP's, from the cavity on x and x's marginal, on the grid
    struct Logit{X}
        x::X
    end
    @define_factor_node(node = Logit, type = Stochastic, interfaces = [:out, :x])
    loglik(y, x) = y == 1 ? -log1p(exp(-x)) : -log1p(exp(x))

    struct ToyFallback end
    # the observation, among the messages or the marginals, as the engine gives a data edge
    observed(node, args) = mean(Dict((MessagePassingRulesBase.rule_inputs(node, args.m)..., MessagePassingRulesBase.rule_inputs(node, args.q)...))[:out])
    function MessagePassingRulesBase.call_fallback(::ToyFallback, node::Type{<:Logit}, target::MessagePassingRulesBase.Target{:x}, algorithm, ctx, args)
        y = observed(node, args)
        return ContinuousUnivariateLogPdf(x -> loglik(y, x))
    end
    function MessagePassingRulesBase.call_energy_fallback(::ToyFallback, node::Type{<:Logit}, algorithm, ctx, args)
        y = observed(node, args)
        ξ, w = weightedmean_precision(args.m[:x])
        h = step(GRID)
        logc = [ξ * x - w * x^2 / 2 for x in GRID]
        integrand = [loglik(y, x) for x in GRID] .+ logc
        top = maximum(integrand)
        logintegral = top + log(sum(exp.(integrand .- top)) * h)
        q = args.q[:x]
        return -logintegral + (ξ * mean(q) - w * (var(q) + mean(q)^2) / 2)
    end
end

@testitem "engine:sites: two likelihoods reach expectation propagation" tags = [:engine] setup = [EngineHarness, SitePolicies] begin
    # The graph of "engine:probit under expectation propagation", with Probit's default algorithm,
    # whose messages are its likelihoods: the sites are computed at the variable, by the policy.
    using ExponentialFamily, Distributions, StandardMessagePassingRules, MessagePassingRulesBase
    H, S = EngineHarness, SitePolicies

    graph = H.Graph()
    w = H.random!(graph)
    y1, y2 = H.data!(graph), H.data!(graph)
    H.node!(graph, NormalMeanVariance, [(:out, w), (:μ, H.constant!(graph, 0.0)), (:v, H.constant!(graph, 1.0))])
    H.node!(graph, Probit, [(:out, y1), (:in, w)]; algorithm = DefaultAlgorithm())
    H.node!(graph, Probit, [(:out, y2), (:in, w)]; algorithm = DefaultAlgorithm())

    events = Any[]
    callbacks = (after_site_computation = (event) -> push!(events, event),)
    policy = S.GridEP()
    result = H.run(graph; data = [y1 => 1.0, y2 => 0.0], iterations = 10, posteriors = [:w => w], free_energy = false, site_policy = policy, variable_callbacks = callbacks)

    # the fixed point, by hand: the test of Probit's own EP rule
    N = Normal()
    sites = [(0.0, 0.0), (0.0, 0.0)]
    for _ in 1:200, i in 1:2
        precision = 1.0 + sum(first, sites) - sites[i][1]
        weighted = sum(last, sites) - sites[i][2]
        mc, vc = weighted / precision, 1 / precision
        sign = i == 1 ? 1.0 : -1.0
        t = sign * mc / sqrt(1 + vc)
        ratio = exp(logpdf(N, t) - logcdf(N, t))
        mt = mc + sign * vc * ratio / sqrt(1 + vc)
        vt = vc - vc^2 * ratio * (t + ratio) / (1 + vc)
        sites[i] = (1 / vt - 1 / vc, mt / vt - mc / vc)
    end
    precision = 1.0 + sum(first, sites)
    q = result.posteriors["w"]
    @test q isa NormalWeightedMeanPrecision
    @test mean(q) ≈ sum(last, sites) / precision atol = 1.0e-7
    @test var(q) ≈ 1 / precision atol = 1.0e-7
    # the events: one per site computed, with the edge; the prior's message is its own site
    @test !isempty(events) && all(event -> event.index in 1:3, events)
    @test any(event -> event.result isa NormalMeanVariance, events)
    @test policy.projections > 0
end

@testitem "engine:sites: closed messages are their own sites, and nothing changes without a policy" tags = [:engine] setup = [EngineHarness, SitePolicies] begin
    # A Gaussian chain observed in Gaussian noise: every message is closed, so the policy projects
    # nothing and the posteriors are the exact ones, as without it.
    using ExponentialFamily, Distributions, StandardMessagePassingRules
    H, S = EngineHarness, SitePolicies

    function chain(T)
        graph = H.Graph()
        x = [H.random!(graph) for _ in 1:T]
        y = [H.data!(graph) for _ in 1:T]
        H.node!(graph, NormalMeanVariance, [(:out, x[1]), (:μ, H.constant!(graph, 0.0)), (:v, H.constant!(graph, 1.0))])
        for t in 2:T
            H.node!(graph, NormalMeanVariance, [(:out, x[t]), (:μ, x[t - 1]), (:v, H.constant!(graph, 0.5))])
        end
        for t in 1:T
            H.node!(graph, NormalMeanVariance, [(:out, y[t]), (:μ, x[t]), (:v, H.constant!(graph, 0.2))])
        end
        return graph, x, y
    end
    data = [0.3, -0.1, 0.8, 0.5, 1.2]
    graph, x, y = chain(5)
    policy = S.GridEP()
    staged = H.run(graph; data = [y => data], iterations = 2, posteriors = [:x => x], site_policy = policy)
    graph, x, y = chain(5)
    plain = H.run(graph; data = [y => data], iterations = 2, posteriors = [:x => x])
    @test policy.projections == 0
    for (a, b) in zip(staged.posteriors["x"], plain.posteriors["x"])
        @test mean(a) ≈ mean(b) && var(a) ≈ var(b)
    end
    @test last(staged.free_energy) ≈ last(plain.free_energy)
end

@testitem "engine:sites: the free energy of a node without rules is expectation propagation's" tags = [:engine] setup = [EngineHarness, SitePolicies] begin
    # x ~ N(0, 1), y ~ Logit(x) observed at 1: a single site, so expectation propagation's evidence
    # is exact, and the free energy is -log ∫ N(x; 0, 1) σ(x) dx, which is log 2 by symmetry.
    using ExponentialFamily, Distributions, StandardMessagePassingRules
    H, S = EngineHarness, SitePolicies

    graph = H.Graph()
    x = H.random!(graph)
    y = H.data!(graph)
    H.node!(graph, NormalMeanVariance, [(:out, x), (:μ, H.constant!(graph, 0.0)), (:v, H.constant!(graph, 1.0))])
    H.node!(graph, S.Logit, [(:out, y), (:x, x)])
    result = H.run(graph; data = [y => 1], iterations = 3, posteriors = [:x => x], site_policy = S.GridEP(), rulefallback = S.ToyFallback())
    @test last(result.free_energy) ≈ log(2.0) atol = 1.0e-6
    q = result.posteriors["x"]
    grid = S.GRID
    p = pdf.(Normal(), grid) .* exp.(S.loglik.(1, grid))
    p ./= sum(p)
    @test mean(q) ≈ sum(p .* grid) atol = 1.0e-6
end
