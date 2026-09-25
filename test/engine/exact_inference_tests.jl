# Whole graphs on which belief propagation is exact: trees of Gaussian, discrete and conjugate
# factors. Every posterior must equal the closed form computed here, independently of the engine,
# and the free energy must equal minus the log evidence, whatever order the engine updates in.

@testitem "engine:iid normals: exact posterior and evidence" tags = [:engine] setup = [EngineHarness] begin
    # x ~ N(0, 10), y[i] ~ N(x, 1): the posterior is conjugate, and y ~ N(0, 10 11ᵀ + I).
    using ExponentialFamily, Distributions, LinearAlgebra, StandardMessagePassingRules
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

    result = H.run(
        graph; data = [y => Y, prior_v => 10.0, v => 1.0], iterations = 2,
        posteriors = [:x => x], annotations = (LogScaleAnnotations(),),
    )
    precision = 1 / 10 + length(Y)
    @test mean(result.posteriors["x"]) ≈ sum(Y) / precision atol = 1.0e-12
    @test var(result.posteriors["x"]) ≈ 1 / precision atol = 1.0e-12
    logevidence = logpdf(MvNormal(zeros(length(Y)), 10 * ones(length(Y), length(Y)) + I), Y)
    @test result.free_energy ≈ fill(-logevidence, 2) atol = 1.0e-9
    # The marginal's log scale is the log evidence too: it is the normaliser of the product.
    @test result.logscales["x"] ≈ logevidence atol = 1.0e-9
end

@testitem "engine:a missing observation is ignored and predicted" tags = [:engine] setup = [EngineHarness] begin
    using ExponentialFamily, Distributions, StandardMessagePassingRules
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

    result = H.run(
        graph; data = [y => Y, prior_v => 10.0, v => 1.0], iterations = 2,
        posteriors = [:x => x], predictions = y, free_energy = false,
    )
    # The posterior is the one of the five observations alone.
    observed = collect(skipmissing(Y))
    precision = 1 / 10 + length(observed)
    q = result.posteriors["x"]
    @test mean(q) ≈ sum(observed) / precision atol = 1.0e-12
    @test var(q) ≈ 1 / precision atol = 1.0e-12
    # A data interface is a cluster of its own, so the message towards the missing y[2] reads
    # q(x): the prediction is N(E[x], v).
    prediction = result.predictions[2]
    @test prediction isa UnivariateNormalDistributionsFamily
    @test mean(prediction) ≈ mean(q) atol = 1.0e-12
    @test var(prediction) ≈ 1.0 atol = 1.0e-12
end

@testitem "engine:gaussian chain: exact smoother and evidence" tags = [:engine] setup = [EngineHarness] begin
    # A random walk x[1] ~ N(0, 10), x[i] ~ N(x[i - 1], 1), observed as y[i] ~ N(x[i], 1). The
    # posteriors are the smoother's, computed here by conditioning the joint Gaussian:
    # Cov(x[i], x[j]) = 10 + min(i, j) - 1 and Cov(y) = Cov(x) + I.
    using ExponentialFamily, Distributions, LinearAlgebra, StandardMessagePassingRules
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

    result = H.run(graph; data = [y => Y, prior_v => 10.0, v => 1.0], iterations = 2, posteriors = [:x => x])
    n = length(Y)
    Σx = [10.0 + min(i, j) - 1 for i in 1:n, j in 1:n]
    Σy = Σx + I
    m = Σx * (Σy \ Y)
    V = Σx - Σx * (Σy \ Σx)
    @test mean.(result.posteriors["x"]) ≈ m atol = 1.0e-10
    @test var.(result.posteriors["x"]) ≈ diag(V) atol = 1.0e-10
    @test result.free_energy ≈ fill(-logpdf(MvNormal(zeros(n), Σy), Y), 2) atol = 1.0e-9
end

@testitem "engine:logic gates: exact marginals by enumeration" tags = [:engine] setup = [EngineHarness] begin
    # x ~ Ber(p), y ~ Ber(0.6), z = x ∧ y, n = ¬z, w ~ Ber(0.4), o = n ∨ w, v ~ Ber(0.5),
    # i = o → v, and a factor Ber(i; 0.9). A tree, so the marginals and the evidence are those
    # of the 16 joint configurations of the free inputs.
    using ExponentialFamily, Distributions, StandardMessagePassingRules
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

    result = H.run(
        graph; data = [p => 0.3], iterations = 2,
        posteriors = [:o => o, :w => w, :n => n, :y => y, :v => v, :z => z, :i => i, :x => x],
    )

    bernoulli(p, b) = b ? p : 1 - p
    names = ("x", "y", "z", "n", "w", "o", "v", "i")
    function enumerate_configurations()
        evidence, when_true = 0.0, Dict(name => 0.0 for name in names)
        for xb in (false, true), yb in (false, true), wb in (false, true), vb in (false, true)
            zb = xb & yb
            nb = !zb
            ob = nb | wb
            ib = !ob | vb
            weight = bernoulli(0.3, xb) * bernoulli(0.6, yb) * bernoulli(0.4, wb) * bernoulli(0.5, vb) * bernoulli(0.9, ib)
            evidence += weight
            for (name, b) in zip(names, (xb, yb, zb, nb, wb, ob, vb, ib))
                b && (when_true[name] += weight)
            end
        end
        return evidence, when_true
    end
    evidence, when_true = enumerate_configurations()
    for name in names
        @test result.posteriors[name] isa Bernoulli
        @test mean(result.posteriors[name]) ≈ when_true[name] / evidence atol = 1.0e-12
    end
    @test result.free_energy ≈ fill(-log(evidence), 2) atol = 1.0e-12
end

@testitem "engine:arithmetic on gaussians: exact posteriors and evidence" tags = [:engine] setup = [EngineHarness] begin
    # s = x + z, d = s - w, m = 2d, y1 ~ N(m, 0.5); v ~ N([0, 1], diag(1, 2)), Av = A v,
    # q = c ⋅ Av, y2 ~ N(q, 0.5). Every variable is a linear map of u = (x, z, w, v), so the
    # posteriors are those of u conditioned on (y1, y2), mapped.
    using ExponentialFamily, Distributions, LinearAlgebra, StandardMessagePassingRules
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

    variables = (; x, z, s, w, d, m, v, Av, q)
    result = H.run(
        graph; data = [y1 => 1.5, y2 => 2.0], iterations = 2,
        posteriors = [name => variables[name] for name in (:w, :m, :d, :s, :v, :Av, :z, :q, :x)],
    )

    # u = (x, z, w, v₁, v₂)
    μ0, V0 = [0.0, 1.0, 0.5, 0.0, 1.0], Diagonal([1.0, 2.0, 1.0, 1.0, 2.0])
    Amat, cvec = [1.0 0.5; 0.0 1.0], [1.0, 2.0]
    maps = Dict(
        "x" => [1.0 0 0 0 0], "z" => [0 1.0 0 0 0], "w" => [0 0 1.0 0 0],
        "s" => [1.0 1.0 0 0 0], "d" => [1.0 1.0 -1.0 0 0], "m" => [2.0 2.0 -2.0 0 0],
        "v" => [zeros(2, 3) I(2)], "Av" => [zeros(2, 3) Amat], "q" => [zeros(1, 3) cvec' * Amat],
    )
    Hobs, R, Yobs = [maps["m"]; maps["q"]], Diagonal([0.5, 0.5]), [1.5, 2.0]
    S = Hobs * V0 * Hobs' + R
    K = V0 * Hobs' / S
    μu, Vu = μ0 + K * (Yobs - Hobs * μ0), V0 - K * S * K'
    for (name, T) in maps
        posterior = result.posteriors[name]
        if size(T, 1) == 1
            @test mean(posterior) ≈ only(T * μu) atol = 1.0e-10
            @test var(posterior) ≈ only(T * Vu * T') atol = 1.0e-10
        else
            @test mean(posterior) ≈ T * μu atol = 1.0e-10
            @test cov(posterior) ≈ T * Vu * T' atol = 1.0e-10
        end
    end
    @test result.free_energy ≈ fill(-logpdf(MvNormal(Hobs * μ0, Matrix(S)), Yobs), 2) atol = 1.0e-9
end

@testitem "engine:beta and uniform priors of bernoulli observations: conjugate posteriors and evidence" tags = [:engine] setup = [EngineHarness] begin
    # p ~ Beta(2, 3) and r ~ Uniform(0, 1) = Beta(1, 1), each with three Bernoulli observations:
    # Beta(a + k, b + n - k), and the evidence B(a + k, b + n - k) / B(a, b).
    using ExponentialFamily, Distributions, SpecialFunctions, StandardMessagePassingRules
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

    result = H.run(
        graph; data = [y => [1.0, 0.0, 1.0], u => [0.0, 0.0, 1.0]], iterations = 2,
        posteriors = [:p => p, :r => r],
    )
    @test all(isapprox.(params(result.posteriors["p"]), (4.0, 4.0); atol = 1.0e-12))
    @test all(isapprox.(params(result.posteriors["r"]), (2.0, 3.0); atol = 1.0e-12))
    logevidence = (logbeta(4, 4) - logbeta(2, 3)) + (logbeta(2, 3) - logbeta(1, 1))
    @test result.free_energy ≈ fill(-logevidence, 2) atol = 1.0e-9
end

@testitem "engine:poisson counts under a gamma prior: conjugate posterior and evidence" tags = [:engine] setup = [EngineHarness] begin
    # l ~ Gamma(shape 2, scale 1.5), y[i] ~ Poisson(l): Gamma(α + Σy, rate β + n), and the
    # evidence βᵅ Γ(α + Σy) / (Γ(α) (β + n)^(α + Σy) Π y!), with β = 1 / 1.5.
    using ExponentialFamily, Distributions, SpecialFunctions, StandardMessagePassingRules
    H = EngineHarness

    graph = H.Graph()
    l = H.random!(graph)
    l_prior = [(:out, l), (:α, H.constant!(graph, 2.0)), (:θ, H.constant!(graph, 1.5))]
    y = [H.data!(graph) for _ in 1:3]
    H.node!(graph, Gamma, l_prior)
    for i in 1:3
        H.node!(graph, Poisson, [(:out, y[i]), (:l, l)])
    end

    Y = [3, 1, 4]
    result = H.run(graph; data = [y => Y], iterations = 2, posteriors = [:l => l])
    α, β = 2.0, 1 / 1.5
    q = result.posteriors["l"]
    @test shape(q) ≈ α + sum(Y) atol = 1.0e-12
    @test rate(q) ≈ β + length(Y) atol = 1.0e-12
    logevidence = α * log(β) - loggamma(α) + loggamma(α + sum(Y)) - (α + sum(Y)) * log(β + length(Y)) - sum(logfactorial, Y)
    @test result.free_energy ≈ fill(-logevidence, 2) atol = 1.0e-9
end

@testitem "engine:inverse-gamma variance: conjugate posterior and evidence" tags = [:engine] setup = [EngineHarness] begin
    # v ~ InverseGamma(3, 2), y[i] ~ N(1, v): InverseGamma(3 + n/2, 2 + S/2), for S = Σ(y - 1)²,
    # and the evidence 2³ Γ(4.5) / (Γ(3) (2 + S/2)^4.5 (2π)^1.5).
    using ExponentialFamily, Distributions, SpecialFunctions, StandardMessagePassingRules
    H = EngineHarness

    graph = H.Graph()
    v = H.random!(graph)
    v_prior = [(:out, v), (:α, H.constant!(graph, 3.0)), (:θ, H.constant!(graph, 2.0))]
    y = [H.data!(graph) for _ in 1:3]
    H.node!(graph, GammaInverse, v_prior)
    for i in 1:3
        H.node!(graph, NormalMeanVariance, [(:out, y[i]), (:μ, H.constant!(graph, 1.0)), (:v, v)])
    end

    Y = [1.2, 0.3, 2.1]
    result = H.run(graph; data = [y => Y], iterations = 2, posteriors = [:v => v])
    S = sum(abs2, Y .- 1.0)
    q = result.posteriors["v"]
    @test mean(q) ≈ mean(InverseGamma(4.5, 2.0 + S / 2)) atol = 1.0e-12
    @test var(q) ≈ var(InverseGamma(4.5, 2.0 + S / 2)) atol = 1.0e-12
    logevidence = 3 * log(2.0) - loggamma(3.0) + loggamma(4.5) - 4.5 * log(2.0 + S / 2) - 1.5 * log(2π)
    @test result.free_energy ≈ fill(-logevidence, 2) atol = 1.0e-9
end

@testitem "engine:half-normal and uninformative factors" tags = [:engine] setup = [EngineHarness] begin
    # h ~ HalfNormal(v0) closed by an Uninformative factor, and u ~ Uninformative with
    # y ~ N(u, 1) observed: q(h) is the prior, q(u) = N(y, 1), and both halves integrate to one,
    # so the free energy is zero.
    using ExponentialFamily, Distributions, StandardMessagePassingRules
    H = EngineHarness

    graph = H.Graph()
    v0, h = H.data!(graph), H.random!(graph)
    u, y = H.random!(graph), H.data!(graph)
    H.node!(graph, HalfNormal, [(:out, h), (:v, v0)])
    H.node!(graph, Uninformative, [(:out, h)])
    H.node!(graph, Uninformative, [(:out, u)])
    H.node!(graph, NormalMeanVariance, [(:out, y), (:μ, u), (:v, H.constant!(graph, 1.0))])

    result = H.run(graph; data = [v0 => 2.0, y => 1.5], iterations = 2, posteriors = [:h => h, :u => u])
    q_h, q_u = result.posteriors["h"], result.posteriors["u"]
    reference = truncated(Normal(0.0, sqrt(2.0)); lower = 0.0)
    @test mean(q_h) ≈ mean(reference) atol = 1.0e-12
    @test var(q_h) ≈ var(reference) atol = 1.0e-12
    @test all(t -> logpdf(q_h, t) ≈ logpdf(reference, t), (0.1, 1.0, 3.0))
    @test mean(q_u) ≈ 1.5 atol = 1.0e-12
    @test var(q_u) ≈ 1.0 atol = 1.0e-12
    @test result.free_energy ≈ [0.0, 0.0] atol = 1.0e-12
end

@testitem "engine:normal-wishart priors from data parameters" tags = [:engine] setup = [EngineHarness] begin
    # MvNormalWishart and MatrixNormalWishart nodes whose parameters are given as data, each
    # variable closed by an Uninformative factor: the posteriors are the priors themselves.
    using ExponentialFamily, Distributions, StandardMessagePassingRules
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

    result = H.run(
        graph; data = [μ0 => [0.5, 1.0], M0 => [1.0 0.5; 0.0 1.0]], iterations = 2,
        posteriors = [:nw => nw, :mw => mw], free_energy = false,
    )
    @test result.posteriors["nw"] == MvNormalWishart([0.5, 1.0], I2, 2.0, 3.0)
    q_mw = result.posteriors["mw"]
    @test q_mw isa MatrixNormalWishart
    @test (q_mw.M, q_mw.U, q_mw.V, q_mw.ν) == ([1.0 0.5; 0.0 1.0], I2, I2, 3.0)
end

@testitem "engine:mixture: exact switch posterior and evidence" tags = [:engine] setup = [EngineHarness] begin
    # s ~ Cat(0.3, 0.7), x[k] ~ N(∓2, 1), z = x[s], y ~ N(z, 0.5) observed at 1.5. Given s = k,
    # y ~ N(μ[k], 1.5), so p(s = k | y) ∝ π[k] N(y; μ[k], 1.5), and that sum is the evidence.
    using ExponentialFamily, StandardMessagePassingRules, Distributions
    import ReactiveMP: LogScaleAnnotations
    H = EngineHarness

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

    result = H.run(
        graph; data = [y => 1.5], iterations = 2,
        posteriors = [:s => s, :x => x], annotations = (LogScaleAnnotations(),), free_energy = false,
    )
    weights = [0.3 * pdf(Normal(-2.0, sqrt(1.5)), 1.5), 0.7 * pdf(Normal(2.0, sqrt(1.5)), 1.5)]
    @test probs(result.posteriors["s"]) ≈ weights ./ sum(weights) atol = 1.0e-12
    @test result.logscales["s"] ≈ log(sum(weights)) atol = 1.0e-10
    # The mixture's message towards an input is its component's likelihood, so q(x[k]) is the
    # posterior of x[k] had y come from component k: precision 1 + 2 and mean (μ[k] + 2 y) / 3.
    for (k, μk) in enumerate((-2.0, 2.0))
        @test mean(result.posteriors["x"][k]) ≈ (μk + 2 * 1.5) / 3 atol = 1.0e-12
        @test var(result.posteriors["x"][k]) ≈ 1 / 3 atol = 1.0e-12
    end
end

@testitem "engine:gaussian coupling: exact joint posterior and evidence" tags = [:engine] setup = [EngineHarness] begin
    # x ~ N(0.5, 1/2), the potential exp(0.5 c x), y ~ N(c, 1) observed at 1.5. The joint of
    # (x, c) is Gaussian with precision [2 -0.5; -0.5 1] and linear term (1, 1.5). The message
    # towards `c` alone is improper; the posterior is not.
    using ExponentialFamily, Distributions, LinearAlgebra, StandardMessagePassingRules, GaussianCouplingMessagePassingRules
    H = EngineHarness

    graph = H.Graph()
    x, c = H.random!(graph), H.random!(graph)
    y = H.data!(graph)
    H.node!(graph, NormalMeanPrecision, [(:out, x), (:μ, H.constant!(graph, 0.5)), (:τ, H.constant!(graph, 2.0))])
    H.node!(graph, GaussianCoupling, [(:out, c), (:in, x), (:a, H.constant!(graph, 0.5))])
    H.node!(graph, NormalMeanVariance, [(:out, y), (:μ, c), (:v, H.constant!(graph, 1.0))])

    result = H.run(graph; data = [y => 1.5], iterations = 3, posteriors = [:x => x, :c => c])
    Λ, h = [2.0 -0.5; -0.5 1.0], [1.0, 1.5]
    Σ = inv(Λ)
    μ = Σ * h
    @test mean(result.posteriors["x"]) ≈ μ[1] atol = 1.0e-12
    @test var(result.posteriors["x"]) ≈ Σ[1, 1] atol = 1.0e-12
    @test mean(result.posteriors["c"]) ≈ μ[2] atol = 1.0e-12
    @test var(result.posteriors["c"]) ≈ Σ[2, 2] atol = 1.0e-12
    # log Z = the factors' constants, -log(π)/2 - 0.5²·2/2 - log(2π)/2 - 1.5²/2, plus the
    # Gaussian integral hᵀΣh/2 + log det(2πΣ)/2.
    logevidence = -log(π) / 2 - 0.25 - log(2π) / 2 - 1.125 + dot(h, Σ * h) / 2 + logdet(2π * Σ) / 2
    @test result.free_energy ≈ fill(-logevidence, 3) atol = 1.0e-9
end

@testmodule StateSpaceModel begin
    # z[i + 1] = A z[i] + B u[i], u[i] ~ N(0, 1), y[i] ~ N(C z[i + 1], 1/10), z[1] ~ N(0, 10⁵ I),
    # four steps; and its posteriors by conditioning the joint Gaussian of w = (z[1], u[1:4]),
    # of which z[i] = T[i] w is a linear map.
    using ExponentialFamily, StandardMessagePassingRules, BIFMMessagePassingRules, LinearAlgebra
    import ..EngineHarness as H

    const A, B, C = [0.9 0.1; 0.0 0.8], reshape([1.0, 0.5], 2, 1), [1.0 0.0]
    const Y = [[0.5], [0.8], [0.3], [-0.1]]
    const I2 = [1.0 0.0; 0.0 1.0]

    function closed_form()
        T = Vector{Matrix{Float64}}(undef, 5)
        T[1] = [I2 zeros(2, 4)]
        for i in 1:4
            T[i + 1] = A * T[i] + B * [zeros(1, 2) (1:4 .== i)']
        end
        V0 = Matrix(Diagonal([1.0e5, 1.0e5, 1.0, 1.0, 1.0, 1.0]))
        Hobs = reduce(vcat, [C * T[i + 1] for i in 1:4])
        S = Hobs * V0 * Hobs' + 0.1 * I
        K = V0 * Hobs' / S
        μw = K * reduce(vcat, Y)
        Vw = V0 - K * S * K'
        return (; T, μw, Vw)
    end

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
        return H.run(graph; data = y .=> Y, iterations = 1, posteriors = [:z => z, :u => u, :yt => yt], free_energy)
    end
end

@testitem "engine:bifm smoother: closed-form posteriors" tags = [:engine] setup = [EngineHarness, StateSpaceModel] begin
    # The prior's variance of 10⁵ amplifies rounding: the covariances reach 88, and agree to 1e-10.
    using ExponentialFamily, BayesBase
    M = StateSpaceModel
    (; T, μw, Vw) = M.closed_form()
    result = M.bifm()
    unwrap(q) = q isa BayesBase.TerminalProdArgument ? q.argument : q
    for i in 1:5
        q = unwrap(result.posteriors["z"][i])
        @test mean(q) ≈ T[i] * μw atol = 1.0e-8
        @test cov(q) ≈ T[i] * Vw * T[i]' atol = 1.0e-8
    end
    for i in 1:4
        q_u, q_yt = unwrap(result.posteriors["u"][i]), unwrap(result.posteriors["yt"][i])
        @test only(mean(q_u)) ≈ μw[2 + i] atol = 1.0e-8
        @test only(cov(q_u)) ≈ Vw[2 + i, 2 + i] atol = 1.0e-8
        CT = M.C * T[i + 1]
        @test mean(q_yt) ≈ CT * μw atol = 1.0e-8
        @test cov(q_yt) ≈ CT * Vw * CT' atol = 1.0e-8
    end
end

@testitem "engine:state space through standard nodes: closed-form posteriors" tags = [:engine] setup = [EngineHarness, StateSpaceModel] begin
    # The same model through `*` and `+`: z[i] = A z[i - 1] + B u[i], y[i] ~ N(C z[i], 1/10).
    # Belief propagation on the chain is the smoother.
    using ExponentialFamily, Distributions, StandardMessagePassingRules
    M, H = StateSpaceModel, EngineHarness
    (; T, μw, Vw) = M.closed_form()

    graph = H.Graph()
    z_prev = H.random!(graph)
    H.node!(graph, MvNormalMeanPrecision, [(:out, z_prev), (:μ, H.constant!(graph, [0.0, 0.0])), (:Λ, H.constant!(graph, [1.0e-5 0.0; 0.0 1.0e-5]))])
    z, u, y = [H.random!(graph) for _ in 1:4], [H.random!(graph) for _ in 1:4], [H.data!(graph) for _ in 1:4]
    for i in 1:4
        H.node!(graph, MvNormalMeanPrecision, [(:out, u[i]), (:μ, H.constant!(graph, [0.0])), (:Λ, H.constant!(graph, [1.0;;]))])
        Az, Bu, Cz = H.random!(graph), H.random!(graph), H.random!(graph)
        H.node!(graph, *, [(:out, Az), (:A, H.constant!(graph, M.A)), (:in, i == 1 ? z_prev : z[i - 1])])
        H.node!(graph, *, [(:out, Bu), (:A, H.constant!(graph, M.B)), (:in, u[i])])
        H.node!(graph, +, [(:out, z[i]), (:in1, Az), (:in2, Bu)])
        H.node!(graph, *, [(:out, Cz), (:A, H.constant!(graph, M.C)), (:in, z[i])])
        H.node!(graph, MvNormalMeanPrecision, [(:out, y[i]), (:μ, Cz), (:Λ, H.constant!(graph, [10.0;;]))])
    end
    result = H.run(graph; data = y .=> M.Y, iterations = 1, posteriors = [:z => [z_prev; z], :u => u], free_energy = false)
    for i in 1:5
        q = result.posteriors["z"][i]
        @test mean(q) ≈ T[i] * μw atol = 1.0e-8
        @test cov(q) ≈ T[i] * Vw * T[i]' atol = 1.0e-7
    end
    for i in 1:4
        @test only(mean(result.posteriors["u"][i])) ≈ μw[2 + i] atol = 1.0e-8
        @test only(cov(result.posteriors["u"][i])) ≈ Vw[2 + i, 2 + i] atol = 1.0e-8
    end
end

@testitem "engine:bifm's free energy is an error" tags = [:engine] setup = [EngineHarness, StateSpaceModel] begin
    # BIFM has no energy: asking for the free energy says so, rather than returning a number.
    using BIFMMessagePassingRules
    @test_throws BIFMMessagePassingRules.BIFMFreeEnergyError StateSpaceModel.bifm(free_energy = true)
end
