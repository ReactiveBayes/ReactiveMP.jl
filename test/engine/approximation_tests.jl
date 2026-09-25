# Whole graphs through nodes whose messages are approximations: a Delta node under the unscented
# transform or linearisation, Probit under expectation propagation, and a normalising flow. The
# approximations are computed again here, by hand, and the posteriors must agree with them.

@testmodule DeltaReference begin
    square_plus_one(x) = x^2 + 1.0
    scaled_square_plus(c, x, s) = c * x^2 + s
    cube_minus(x) = x^3 - x

    # The unscented transform of N(m, v) through g, with the defaults of `Unscented()`
    # (α = 1e-3, β = 2, κ = 0): the mean and variance of g(x), and the covariance of x and g(x).
    function unscented(g, m, v; α = 1.0e-3, β = 2.0, κ = 0.0)
        λ = α^2 * (1 + κ) - 1
        χ = [m, m + sqrt((1 + λ) * v), m - sqrt((1 + λ) * v)]
        Wm = [λ / (1 + λ), 1 / (2 * (1 + λ)), 1 / (2 * (1 + λ))]
        Wc = Wm .+ [1 - α^2 + β, 0.0, 0.0]
        Y = g.(χ)
        μ = sum(Wm .* Y)
        return μ, sum(Wc .* (Y .- μ) .^ 2), sum(Wc .* (χ .- m) .* (Y .- μ))
    end

    # x ~ N(mx, vx), z = g(x) through a joint Gaussian of mean (mx, mz) and covariance
    # [vx C; C vz], y ~ N(z, r) observed: q(z) is the product of N(mz, vz) with N(y, r), and
    # q(x) is x conditioned on y in the joint.
    function posteriors(mx, vx, mz, vz, C, y, r)
        pz = 1 / vz + 1 / r
        K = C / (vz + r)
        return (mean_z = (mz / vz + y / r) / pz, var_z = 1 / pz, mean_x = mx + K * (y - mz), var_x = vx - K * C)
    end

    # The Bethe free energy of prior → deterministic node → likelihood: the prior's energy less
    # q(x)'s entropy (the node's term is -H[q(x)], which x's degree adds back once), and the
    # likelihood's energy E[-log N(y; z, r)].
    function free_energy(mx0, vx0, q, y, r)
        prior = (log(2π * vx0) + ((q.mean_x - mx0)^2 + q.var_x) / vx0) / 2
        entropy = log(2π * ℯ * q.var_x) / 2
        likelihood = (log(2π * r) + ((y - q.mean_z)^2 + q.var_z) / r) / 2
        return prior - entropy + likelihood
    end
end

@testitem "engine:delta node under the unscented transform" tags = [:engine] setup = [EngineHarness, DeltaReference] begin
    # x ~ N(0.5, 1), z = x² + 1, y ~ N(z, 0.1) observed at 2. The sigma points' weights reach
    # 1e6 in magnitude, so the hand transform and the rule round differently, below 1e-9.
    using ExponentialFamily, Distributions, StandardMessagePassingRules, DeltaMessagePassingRules, MessagePassingRulesApproximations
    H, R = EngineHarness, DeltaReference
    f = R.square_plus_one

    graph = H.Graph()
    x = H.random!(graph)
    prior = [(:out, x), (:μ, H.constant!(graph, 0.5)), (:v, H.constant!(graph, 1.0))]
    z = H.random!(graph)
    y = H.data!(graph)
    H.node!(graph, NormalMeanVariance, prior)
    H.node!(graph, DeltaFn{typeof(f)}, [(:out, z), ((:in, 1), x)]; algorithm = DeltaApproximation(method = Unscented()), nodefn = f)
    H.node!(graph, NormalMeanVariance, [(:out, y), (:μ, z), (:v, H.constant!(graph, 0.1))])

    result = H.run(
        graph; data = [y => 2.0], iterations = 3,
        posteriors = [:z => z, :x => x], initial_marginals = [z => NormalMeanVariance(1.0, 1.0)],
    )
    q = R.posteriors(0.5, 1.0, R.unscented(f, 0.5, 1.0)..., 2.0, 0.1)
    @test mean(result.posteriors["z"]) ≈ q.mean_z atol = 1.0e-9
    @test var(result.posteriors["z"]) ≈ q.var_z atol = 1.0e-9
    @test mean(result.posteriors["x"]) ≈ q.mean_x atol = 1.0e-9
    @test var(result.posteriors["x"]) ≈ q.var_x atol = 1.0e-9
    @test result.free_energy ≈ fill(R.free_energy(0.5, 1.0, q, 2.0, 0.1), 3) atol = 1.0e-9
end

@testitem "engine:delta node under linearisation" tags = [:engine] setup = [EngineHarness, DeltaReference] begin
    # x ~ N(0.5, 1), z = x³ - x, y ~ N(z, 0.1) observed at 2: f linearised at the prior mean,
    # f(0.5) = -0.375 and f'(0.5) = -0.25, makes (x, z) jointly Gaussian.
    using ExponentialFamily, Distributions, StandardMessagePassingRules, DeltaMessagePassingRules, MessagePassingRulesApproximations
    H, R = EngineHarness, DeltaReference
    f = R.cube_minus

    graph = H.Graph()
    x, z = H.random!(graph), H.random!(graph)
    y = H.data!(graph)
    H.node!(graph, NormalMeanVariance, [(:out, x), (:μ, H.constant!(graph, 0.5)), (:v, H.constant!(graph, 1.0))])
    H.node!(graph, DeltaFn{typeof(f)}, [(:out, z), ((:in, 1), x)]; algorithm = DeltaApproximation(method = Linearization()), nodefn = f)
    H.node!(graph, NormalMeanVariance, [(:out, y), (:μ, z), (:v, H.constant!(graph, 0.1))])

    result = H.run(
        graph; data = [y => 2.0], iterations = 3,
        posteriors = [:z => z, :x => x], initial_marginals = [z => NormalMeanVariance(1.0, 1.0)],
    )
    J = 3 * 0.5^2 - 1
    q = R.posteriors(0.5, 1.0, f(0.5), J^2, J, 2.0, 0.1)
    @test mean(result.posteriors["z"]) ≈ q.mean_z atol = 1.0e-12
    @test var(result.posteriors["z"]) ≈ q.var_z atol = 1.0e-12
    @test mean(result.posteriors["x"]) ≈ q.mean_x atol = 1.0e-12
    @test var(result.posteriors["x"]) ≈ q.var_x atol = 1.0e-12
    @test result.free_energy ≈ fill(R.free_energy(0.5, 1.0, q, 2.0, 0.1), 3) atol = 1.0e-12
end

@testitem "engine:delta node with static inputs folded into its function" tags = [:engine] setup = [EngineHarness, DeltaReference] begin
    # z = f(2.0, x, s), with the constant 2.0 and the data s = 1 folded into the function, so x
    # is the node's only input: the posteriors are those of z = 2x² + 1 under the unscented
    # transform, and y ~ N(z, 0.1) is observed at 3.
    using ExponentialFamily, Distributions, StandardMessagePassingRules, DeltaMessagePassingRules, MessagePassingRulesApproximations
    import ReactiveMP: getinterfaces
    H, R = EngineHarness, DeltaReference
    f = R.scaled_square_plus

    graph = H.Graph()
    x = H.random!(graph)
    prior = [(:out, x), (:μ, H.constant!(graph, 0.5)), (:v, H.constant!(graph, 1.0))]
    z = H.random!(graph)
    c = H.constant!(graph, 2.0)
    s = H.data!(graph)
    y = H.data!(graph)
    H.node!(graph, NormalMeanVariance, prior)
    delta = H.node!(graph, DeltaFn{typeof(f)}, [(:out, z), ((:in, 1), c), ((:in, 2), x), ((:in, 3), s)]; algorithm = DeltaApproximation(method = Unscented()), nodefn = f)
    H.node!(graph, NormalMeanVariance, [(:out, y), (:μ, z), (:v, H.constant!(graph, 0.1))])

    result = H.run(
        graph; data = [y => 3.0, s => 1.0], iterations = 3,
        posteriors = [:z => z, :x => x], initial_marginals = [z => NormalMeanVariance(1.0, 1.0)],
    )
    @test length(getinterfaces(delta)) == 2
    folded(x) = f(2.0, x, 1.0)
    q = R.posteriors(0.5, 1.0, R.unscented(folded, 0.5, 1.0)..., 3.0, 0.1)
    @test mean(result.posteriors["z"]) ≈ q.mean_z atol = 1.0e-9
    @test var(result.posteriors["z"]) ≈ q.var_z atol = 1.0e-9
    @test mean(result.posteriors["x"]) ≈ q.mean_x atol = 1.0e-9
    @test var(result.posteriors["x"]) ≈ q.var_x atol = 1.0e-9
    @test result.free_energy ≈ fill(R.free_energy(0.5, 1.0, q, 3.0, 0.1), 3) atol = 1.0e-9
end

@testitem "engine:probit under expectation propagation" tags = [:engine] setup = [EngineHarness] begin
    # w ~ N(0, 1), y1 ~ Ber(Φ(w)) observed at 1 and y2 at 0. Expectation propagation run by hand
    # to its fixed point: each site's cavity, the moments of its tilted distribution in closed
    # form, and the site as their quotient. Each Probit's rule reads the message on its own edge,
    # which the other's feeds, so the engine iterates towards the same fixed point.
    using ExponentialFamily, Distributions, StandardMessagePassingRules, ProbitMessagePassingRules
    H = EngineHarness

    graph = H.Graph()
    w = H.random!(graph)
    y1, y2 = H.data!(graph), H.data!(graph)
    H.node!(graph, NormalMeanVariance, [(:out, w), (:μ, H.constant!(graph, 0.0)), (:v, H.constant!(graph, 1.0))])
    H.node!(graph, Probit, [(:out, y1), (:in, w)])
    H.node!(graph, Probit, [(:out, y2), (:in, w)])

    result = H.run(graph; data = [y1 => 1.0, y2 => 0.0], iterations = 10, posteriors = [:w => w])

    N = Normal()
    function expectation_propagation(ys; sweeps = 100)
        sites = [(0.0, 0.0) for _ in ys] # (precision, weighted mean)
        for _ in 1:sweeps, i in eachindex(ys)
            precision = 1.0 + sum(first, sites) - sites[i][1]
            weighted = sum(last, sites) - sites[i][2]
            mc, vc = weighted / precision, 1 / precision
            sign = ys[i] == 1 ? 1.0 : -1.0
            t = sign * mc / sqrt(1 + vc)
            ratio = exp(logpdf(N, t) - logcdf(N, t))
            mt = mc + sign * vc * ratio / sqrt(1 + vc)
            vt = vc - vc^2 * ratio * (t + ratio) / (1 + vc)
            sites[i] = (1 / vt - 1 / vc, mt / vt - mc / vc)
        end
        precision = 1.0 + sum(first, sites)
        return sum(last, sites) / precision, 1 / precision
    end
    m, v = expectation_propagation([1, 0])
    q = result.posteriors["w"]
    @test mean(q) ≈ m atol = 1.0e-10
    @test var(q) ≈ v atol = 1.0e-10

    # On this tree the free energy is E_q[-log N(w; 0, 1) - log Φ(w) - log Φ(-w)] - H[q], here
    # by the trapezoidal rule on [-12, 12]; the engine's cubature of log Φ agrees to 1e-12.
    grid = range(-12.0, 12.0; length = 200_001)
    density = pdf.(Normal(m, sqrt(v)), grid)
    energy = sum(density .* (-logpdf.(N, grid) .- logcdf.(N, grid) .- logccdf.(N, grid))) * step(grid)
    bethe = energy - entropy(Normal(m, sqrt(v)))
    @test last(result.free_energy) ≈ bethe atol = 1.0e-10
    # and it bounds -log p(y) from above; expectation propagation is close to it here.
    evidence = sum(pdf.(N, grid) .* cdf.(N, grid) .* ccdf.(N, grid)) * step(grid)
    @test -log(evidence) <= last(result.free_energy) <= -log(evidence) + 1.0e-4
end

@testmodule FlowReference begin
    # x[k] ~ N(z_μ, z_Λ⁻¹), y_lat[k] = f(x[k]), y[k] ~ N(y_lat[k], 0.01 I), under mean-field over
    # x, z_μ and z_Λ, with f a small normalising flow; and the same mean-field fixed point
    # computed by hand, from the messages the flow sends back towards each x[k].
    using ExponentialFamily, Distributions, LinearAlgebra, StandardMessagePassingRules, FlowMessagePassingRules
    import ..EngineHarness as H

    const MODEL = compile(
        FlowModel((InputLayer(2), AdditiveCouplingLayer(PlanarFlow(); permute = false), PermutationLayer(PermutationMatrix([2, 1])), AdditiveCouplingLayer(PlanarFlow(); permute = false))),
        [0.3, -0.2, 0.5, 0.1, 0.4, -0.3],
    )
    const Y = [[1.2, 0.3], [0.8, -0.4], [1.5, 0.9], [0.2, 0.1]]
    const I2 = [1.0 0.0; 0.0 1.0]
    const NOISE = 0.01 * I2

    function run(method; iterations)
        graph = H.Graph()
        z_μ, z_Λ = H.random!(graph), H.random!(graph)
        H.node!(graph, MvNormalMeanCovariance, [(:out, z_μ), (:μ, H.constant!(graph, [0.0, 0.0])), (:Σ, H.constant!(graph, 100 * I2))])
        H.node!(graph, Wishart, [(:out, z_Λ), (:ν, H.constant!(graph, 3.0)), (:S, H.constant!(graph, I2))])
        x, y_lat, y = [H.random!(graph) for _ in 1:4], [H.random!(graph) for _ in 1:4], [H.data!(graph) for _ in 1:4]
        for k in 1:4
            H.node!(graph, MvNormalMeanPrecision, [(:out, x[k]), (:μ, z_μ), (:Λ, z_Λ)]; factorisation = ((:out,), (:μ,), (:Λ,)))
            H.node!(graph, Flow, [(:out, y_lat[k]), (:in, x[k])]; algorithm = FlowApproximation(MODEL; method))
            H.node!(graph, MvNormalMeanCovariance, [(:out, y[k]), (:μ, y_lat[k]), (:Σ, H.constant!(graph, NOISE))])
        end
        return H.run(
            graph; data = y .=> Y, iterations, posteriors = [:z_μ => z_μ, :z_Λ => z_Λ, :x => x],
            initial_marginals = [z_μ => MvNormalMeanCovariance([0.0, 0.0], 100 * I2), z_Λ => Wishart(3.0, I2)],
        )
    end

    inverse(y) = FlowMessagePassingRules.backward(MODEL, y)

    # Linearisation: N(f⁻¹(y), J Σ Jᵀ), J the Jacobian of f⁻¹ at y, here by central differences.
    function linearised(y)
        h = 1.0e-6
        J = reduce(hcat, [(inverse(y .+ h .* I2[:, j]) .- inverse(y .- h .* I2[:, j])) ./ (2h) for j in 1:2])
        return inverse(y), J * NOISE * J'
    end

    # The unscented transform of N(y, Σ) through f⁻¹, with `Unscented(2)`'s defaults (α = 1e-3,
    # β = 2, κ = 0), its sigma points along the rows of the symmetric root of (L + λ) Σ.
    function unscented(y; α = 1.0e-3, β = 2.0, κ = 0.0)
        L = 2
        λ = α^2 * (L + κ) - L
        root = sqrt(Symmetric((L + λ) * NOISE))
        χ = [y, (y .+ root[l, :] for l in 1:L)..., (y .- root[l, :] for l in 1:L)...]
        Wm = [λ / (L + λ); fill(1 / (2 * (L + λ)), 2L)]
        Wc = copy(Wm)
        Wc[1] += 1 - α^2 + β
        images = inverse.(χ)
        μ = sum(Wm .* images)
        return μ, sum(Wc[k] * (images[k] - μ) * (images[k] - μ)' for k in eachindex(χ))
    end

    # Mean-field coordinate ascent, from the messages (mean, covariance) towards each x[k]:
    # q(x[k]) ∝ message × N(E[z_μ], E[z_Λ]⁻¹), q(z_μ) = N(0, 100 I) × Π N(E[x[k]], E[z_Λ]⁻¹),
    # q(z_Λ) = Wishart(3 + 4, (I + Σ E[(x[k] - z_μ)(x[k] - z_μ)ᵀ])⁻¹).
    function coordinate_ascent(messages; sweeps = 500)
        Eμ, Vμ, S = zeros(2), 100 * I2, I2
        mx, Vx = [zeros(2) for _ in 1:4], [I2 for _ in 1:4]
        for _ in 1:sweeps
            EΛ = 7 * S
            Vx = [inv(inv(Σk) + EΛ) for (_, Σk) in messages]
            mx = [Vx[k] * (inv(messages[k][2]) * messages[k][1] + EΛ * Eμ) for k in 1:4]
            Vμ = inv(I2 / 100 + 4 * EΛ)
            Eμ = Vμ * (EΛ * sum(mx))
            S = inv(I2 + sum(Vx[k] + Vμ + (mx[k] - Eμ) * (mx[k] - Eμ)' for k in 1:4))
        end
        return (; Eμ, Vμ, EΛ = 7 * S, mx, Vx)
    end
end

@testitem "engine:normalising flow under linearisation" tags = [:engine] setup = [EngineHarness, FlowReference] begin
    # The finite differences are accurate to 1e-10; the fixed points agree to 1e-9.
    using ExponentialFamily, Distributions
    import MessagePassingRulesApproximations as A
    F = FlowReference
    result = F.run(A.Linearization(); iterations = 40)
    reference = F.coordinate_ascent(F.linearised.(F.Y))

    @test mean(result.posteriors["z_μ"]) ≈ reference.Eμ atol = 1.0e-9
    @test cov(result.posteriors["z_μ"]) ≈ reference.Vμ atol = 1.0e-9
    @test result.posteriors["z_Λ"] isa Wishart
    @test mean(result.posteriors["z_Λ"]) ≈ reference.EΛ atol = 1.0e-9
    for k in 1:4
        @test mean(result.posteriors["x"][k]) ≈ reference.mx[k] atol = 1.0e-9
        @test cov(result.posteriors["x"][k]) ≈ reference.Vx[k] atol = 1.0e-9
    end
    @test all(isfinite, result.free_energy)
    @test abs(result.free_energy[end] - result.free_energy[end - 1]) < 1.0e-10
end

@testitem "engine:normalising flow under the unscented transform" tags = [:engine] setup = [EngineHarness, FlowReference] begin
    # The sigma points' weights reach 1e6 in magnitude; the fixed points agree to 1e-8.
    using ExponentialFamily, Distributions
    import MessagePassingRulesApproximations as A
    F = FlowReference
    result = F.run(A.Unscented(2); iterations = 40)
    reference = F.coordinate_ascent(F.unscented.(F.Y))

    @test mean(result.posteriors["z_μ"]) ≈ reference.Eμ atol = 1.0e-8
    @test cov(result.posteriors["z_μ"]) ≈ reference.Vμ atol = 1.0e-8
    @test mean(result.posteriors["z_Λ"]) ≈ reference.EΛ atol = 1.0e-8
    for k in 1:4
        @test mean(result.posteriors["x"][k]) ≈ reference.mx[k] atol = 1.0e-8
        @test cov(result.posteriors["x"][k]) ≈ reference.Vx[k] atol = 1.0e-8
    end
    @test all(isfinite, result.free_energy)
    @test abs(result.free_energy[end] - result.free_energy[end - 1]) < 1.0e-10
end
