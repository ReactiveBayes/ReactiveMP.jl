# Whole graphs under variational message passing. The schedule is the engine's own and may
# change, so no test pins the path: each checks the fixed point the engine settles at. Where the
# model is conjugate, the fixed point is computed again here by coordinate ascent, or checked by
# one sweep of hand-derived updates from the engine's posteriors, which must give them back.
# The free energy must be finite and settle; it must not increase where the schedule guarantees
# it: with two blocks of variables each update is an exact coordinate step, whatever the order.

@testmodule VariationalChecks begin
    # The free energy has settled: its last change is below `tol`.
    settled(energies; tol = 1.0e-10) = length(energies) >= 2 && all(isfinite, energies) && abs(energies[end] - energies[end - 1]) < tol
    # It never increases, up to rounding.
    nonincreasing(energies; tol = 1.0e-10) = all(<=(tol), diff(energies))
end

@testitem "engine:mean-field normal-gamma: coordinate ascent and its free energy" tags = [:engine] setup = [EngineHarness, VariationalChecks] begin
    # μ ~ N(0, 1/0.01), τ ~ Gamma(1, 1), y[i] ~ N(μ, 1/τ) under q(μ)q(τ):
    # q(μ) = N with precision 0.01 + n E[τ] and weighted mean E[τ] Σy, and
    # q(τ) = Gamma(1 + n/2, 1 + Σ((y - E[μ])² + Var[μ])/2).
    using ExponentialFamily, Distributions, SpecialFunctions, StandardMessagePassingRules
    H, V = EngineHarness, VariationalChecks

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

    result = H.run(
        graph; data = [y => Y], iterations = 20,
        posteriors = [:μ => μ, :τ => τ], initial_marginals = [τ => GammaShapeRate(1.0, 1.0)],
    )

    n = length(Y)
    m, v, a, b = let
        m, v, a, b = 0.0, 100.0, 1.0, 1.0
        for _ in 1:200
            Eτ = a / b
            v = 1 / (0.01 + n * Eτ)
            m = v * Eτ * sum(Y)
            a, b = 1 + n / 2, 1 + sum((Y .- m) .^ 2 .+ v) / 2
        end
        (m, v, a, b)
    end
    q_μ, q_τ = result.posteriors["μ"], result.posteriors["τ"]
    @test mean(q_μ) ≈ m atol = 1.0e-10
    @test var(q_μ) ≈ v atol = 1.0e-10
    @test shape(q_τ) ≈ a atol = 1.0e-10
    @test rate(q_τ) ≈ b atol = 1.0e-10

    # F = E[-log p(μ)] + E[-log p(τ)] + Σ E[-log p(y[i] | μ, τ)] - H[q(μ)] - H[q(τ)].
    Eτ, Elogτ = a / b, digamma(a) - log(b)
    prior_μ = (log(2π / 0.01) + 0.01 * (m^2 + v)) / 2
    prior_τ = Eτ # Gamma(1, 1): -log p(τ) = τ
    likelihood = sum((log(2π) - Elogτ + Eτ * ((yi - m)^2 + v)) / 2 for yi in Y)
    F = prior_μ + prior_τ + likelihood - entropy(Normal(m, sqrt(v))) - entropy(Gamma(a, 1 / b))
    @test last(result.free_energy) ≈ F atol = 1.0e-10
    @test V.nonincreasing(result.free_energy)
    @test V.settled(result.free_energy)
end

@testitem "engine:structured normal-gamma: coordinate ascent" tags = [:engine] setup = [EngineHarness, VariationalChecks] begin
    # μ ~ N(0, 1/0.01), τ ~ Gamma(1, 1), x[i] ~ N(μ, 1/τ), y[i] ~ N(x[i], 0.5), under
    # q(x[i], μ)q(τ): given E[τ], belief propagation over the star of μ and the x is exact, so the
    # Gaussian part is the joint of (μ, x) with precision E[τ] on each x[i] - μ, and
    # q(τ) = Gamma(1 + n/2, 1 + Σ E[(x[i] - μ)²]/2). The free energy mixes belief propagation with
    # the variational update of τ, and need not decrease along the way.
    using ExponentialFamily, Distributions, LinearAlgebra, StandardMessagePassingRules
    H, V = EngineHarness, VariationalChecks

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
        H.node!(graph, NormalMeanPrecision, [(:out, x[i]), (:μ, μ), (:τ, τ)]; factorisation = ((:out, :μ), (:τ,)))
        H.node!(graph, NormalMeanVariance, observations[i])
    end

    result = H.run(
        graph; data = [y => Y], iterations = 40,
        posteriors = [:μ => μ, :τ => τ, :x => x],
        initial_marginals = [τ => GammaShapeRate(1.0, 1.0), μ => NormalMeanPrecision(0.0, 1.0)],
    )

    # The joint of (μ, x[1:n]), μ first.
    n = length(Y)
    a, b, m, Σ = let
        a, b = 1.0, 1.0
        local m, Σ
        for _ in 1:500
            t = a / b
            Λ = zeros(n + 1, n + 1)
            Λ[1, 1] = 0.01 + n * t
            for i in 1:n
                Λ[1 + i, 1 + i] = t + 2.0
                Λ[1, 1 + i] = Λ[1 + i, 1] = -t
            end
            Σ = inv(Λ)
            m = Σ * [0.0; 2.0 .* Y]
            spread = sum((m[1 + i] - m[1])^2 + Σ[1 + i, 1 + i] + Σ[1, 1] - 2 * Σ[1, 1 + i] for i in 1:n)
            a, b = 1 + n / 2, 1 + spread / 2
        end
        (a, b, m, Σ)
    end
    @test mean(result.posteriors["μ"]) ≈ m[1] atol = 1.0e-9
    @test var(result.posteriors["μ"]) ≈ Σ[1, 1] atol = 1.0e-9
    @test shape(result.posteriors["τ"]) ≈ a atol = 1.0e-9
    @test rate(result.posteriors["τ"]) ≈ b atol = 1.0e-9
    @test mean.(result.posteriors["x"]) ≈ m[2:end] atol = 1.0e-9
    @test var.(result.posteriors["x"]) ≈ diag(Σ)[2:end] atol = 1.0e-9
    @test V.settled(result.free_energy)
end

@testitem "engine:normal mixture: a fixed point of coordinate ascent" tags = [:engine] setup = [EngineHarness, VariationalChecks] begin
    # Two components under mean-field, π ~ Dir(1, 1), m[k] ~ N(∓1, 1/0.1), p[k] ~ Gamma(1, 1),
    # z[i] ~ Cat(π), y[i] ~ N(m[z[i]], 1/p[z[i]]). From the engine's posteriors, one sweep of the
    # conjugate updates must give them back, and the clusters are the data's two signs.
    using ExponentialFamily, Distributions, SpecialFunctions, StandardMessagePassingRules
    H, V = EngineHarness, VariationalChecks

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

    result = H.run(
        graph; data = [y => Y], iterations = 20,
        posteriors = [:m => m, :π => π, :p => p, :z => z],
        initial_marginals = [
            π => Dirichlet([1.0, 1.0]), m[1] => NormalMeanPrecision(-1.0, 0.1), m[2] => NormalMeanPrecision(1.0, 0.1),
            p[1] => GammaShapeRate(1.0, 1.0), p[2] => GammaShapeRate(1.0, 1.0),
        ],
    )
    q_π, q_m, q_p, q_z = result.posteriors["π"], result.posteriors["m"], result.posteriors["p"], result.posteriors["z"]
    r = reduce(vcat, [probs(q)' for q in q_z]) # r[i, k] = q(z[i] = k)
    α = q_π.alpha
    Elogπ = digamma.(α) .- digamma(sum(α))
    Ep, Elogp = shape.(q_p) ./ rate.(q_p), digamma.(shape.(q_p)) .- log.(rate.(q_p))
    Em, Vm = mean.(q_m), var.(q_m)
    spread(i, k) = (Y[i] - Em[k])^2 + Vm[k]

    @test α ≈ [1.0, 1.0] .+ vec(sum(r; dims = 1)) atol = 1.0e-9
    for (k, μ0) in enumerate((-1.0, 1.0))
        λk = 0.1 + Ep[k] * sum(r[:, k])
        @test precision(q_m[k]) ≈ λk atol = 1.0e-9
        @test mean(q_m[k]) ≈ (0.1 * μ0 + Ep[k] * sum(r[:, k] .* Y)) / λk atol = 1.0e-9
        @test shape(q_p[k]) ≈ 1 + sum(r[:, k]) / 2 atol = 1.0e-9
        @test rate(q_p[k]) ≈ 1 + sum(r[i, k] * spread(i, k) for i in eachindex(Y)) / 2 atol = 1.0e-9
    end
    for i in eachindex(Y)
        logits = [Elogπ[k] + Elogp[k] / 2 - Ep[k] * spread(i, k) / 2 for k in 1:2]
        @test r[i, :] ≈ exp.(logits .- maximum(logits)) ./ sum(exp.(logits .- maximum(logits))) atol = 1.0e-9
        @test argmax(r[i, :]) == (Y[i] < 0 ? 1 : 2)
    end
    @test V.settled(result.free_energy)
end

@testitem "engine:gamma mixture: a fixed point of coordinate ascent" tags = [:engine] setup = [EngineHarness, VariationalChecks] begin
    # Two components under mean-field, the shapes a = (2, 5) given as data and the rates learned:
    # π ~ Dir(1, 1), b[k] ~ Gamma(α[k], β[k]), z[i] ~ Cat(π), y[i] ~ Gamma(a[z[i]], rate b[z[i]]).
    # One sweep of the conjugate updates from the engine's posteriors gives them back. It
    # converges slowly: by 80 iterations the free energy changes by less than 1e-12.
    using ExponentialFamily, Distributions, SpecialFunctions, StandardMessagePassingRules
    H, V = EngineHarness, VariationalChecks
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

    result = H.run(
        graph; data = [y => Y, a1 => 2.0, a2 => 5.0], iterations = 80,
        posteriors = [:b => b, :π => π, :z => z],
        initial_marginals = [π => Dirichlet([1.0, 1.0]), b[1] => GammaShapeRate(1.0, 1.0), b[2] => GammaShapeRate(1.0, 1.0)],
    )
    a, α0, β0 = [2.0, 5.0], [2.0, 2.0], [4.0, 1.0]
    q_π, q_b, q_z = result.posteriors["π"], result.posteriors["b"], result.posteriors["z"]
    r = reduce(vcat, [probs(q)' for q in q_z])
    Elogπ = digamma.(q_π.alpha) .- digamma(sum(q_π.alpha))
    Eb, Elogb = shape.(q_b) ./ rate.(q_b), digamma.(shape.(q_b)) .- log.(rate.(q_b))

    @test q_π.alpha ≈ [1.0, 1.0] .+ vec(sum(r; dims = 1)) atol = 1.0e-8
    for k in 1:2
        @test shape(q_b[k]) ≈ α0[k] + a[k] * sum(r[:, k]) atol = 1.0e-8
        @test rate(q_b[k]) ≈ β0[k] + sum(r[:, k] .* Y) atol = 1.0e-8
    end
    for i in eachindex(Y)
        logits = [Elogπ[k] + a[k] * Elogb[k] - loggamma(a[k]) + (a[k] - 1) * log(Y[i]) - Eb[k] * Y[i] for k in 1:2]
        @test r[i, :] ≈ exp.(logits .- maximum(logits)) ./ sum(exp.(logits .- maximum(logits))) atol = 1.0e-8
    end
    @test V.settled(result.free_energy; tol = 1.0e-12)
end

@testitem "engine:softdot regression: coordinate ascent" tags = [:engine] setup = [EngineHarness, VariationalChecks] begin
    # θ ~ N(0, I), γ ~ Gamma(2, 1), y[i] ~ N(θ ⋅ X[i], 1/γ) under mean-field:
    # q(θ) = N with precision I + E[γ] Σ X Xᵀ and weighted mean E[γ] Σ y X, and
    # q(γ) = Gamma(2 + n/2, 1 + Σ((y - E[θ] ⋅ X)² + Xᵀ Cov[θ] X)/2).
    using ExponentialFamily, Distributions, LinearAlgebra, StandardMessagePassingRules, SoftDotMessagePassingRules
    H, V = EngineHarness, VariationalChecks
    node!(graph, fform, interfaces) = H.node!(graph, fform, interfaces; factorisation = H.meanfield_factorisation(interfaces))

    graph = H.Graph()
    θ, γ = H.random!(graph), H.random!(graph)
    node!(graph, MvNormalMeanPrecision, [(:out, θ), (:μ, H.constant!(graph, [0.0, 0.0])), (:Λ, H.constant!(graph, [1.0 0.0; 0.0 1.0]))])
    node!(graph, GammaShapeRate, [(:out, γ), (:α, H.constant!(graph, 2.0)), (:β, H.constant!(graph, 1.0))])
    ys, Xs = [H.data!(graph) for _ in 1:3], [H.data!(graph) for _ in 1:3]
    for (y, X) in zip(ys, Xs)
        node!(graph, SoftDot, [(:y, y), (:θ, θ), (:x, X), (:γ, γ)])
    end

    X, Y = [[1.0, 0.5], [0.3, -1.0], [2.0, 1.0]], [1.2, -0.4, 2.1]
    result = H.run(
        graph; data = [(Xs .=> X)..., (ys .=> Y)...], iterations = 20, posteriors = [:γ => γ, :θ => θ],
        initial_marginals = [θ => MvNormalMeanPrecision([0.0, 0.0], [1.0 0.0; 0.0 1.0]), γ => GammaShapeRate(2.0, 1.0)],
    )

    a, b, m, Σ = let
        a, b = 2.0, 1.0
        local m, Σ
        for _ in 1:200
            Eγ = a / b
            Σ = inv(I + Eγ * sum(x * x' for x in X))
            m = Σ * (Eγ * sum(Y .* X))
            a, b = 2 + length(Y) / 2, 1 + sum((Y[i] - dot(m, X[i]))^2 + dot(X[i], Σ * X[i]) for i in eachindex(Y)) / 2
        end
        (a, b, m, Σ)
    end
    @test mean(result.posteriors["θ"]) ≈ m atol = 1.0e-10
    @test cov(result.posteriors["θ"]) ≈ Σ atol = 1.0e-10
    @test shape(result.posteriors["γ"]) ≈ a atol = 1.0e-10
    @test rate(result.posteriors["γ"]) ≈ b atol = 1.0e-10
    @test V.nonincreasing(result.free_energy)
    @test V.settled(result.free_energy)
end

@testitem "engine:matrix-normal covariances: coordinate ascent" tags = [:engine] setup = [EngineHarness, VariationalChecks] begin
    # U ~ IW(5, I), V ~ IW(4, 2I), y[i] ~ MN(M, U, V) under mean-field: with R[i] = y[i] - M,
    # q(U) = IW(5 + 3·2, I + Σ R E[V⁻¹] Rᵀ) and q(V) = IW(4 + 3·2, 2I + Σ Rᵀ E[U⁻¹] R), and
    # E[U⁻¹] = ν Ψ⁻¹ for IW(ν, Ψ).
    using ExponentialFamily, Distributions, LinearAlgebra, StandardMessagePassingRules
    H, V = EngineHarness, VariationalChecks
    meanfield!(graph, fform, interfaces) = H.node!(graph, fform, interfaces; factorisation = H.meanfield_factorisation(interfaces))
    I2 = [1.0 0.0; 0.0 1.0]

    graph = H.Graph()
    U = H.random!(graph)
    U_prior = [(:out, U), (:ν, H.constant!(graph, 5.0)), (:S, H.constant!(graph, I2))]
    Vv = H.random!(graph)
    V_prior = [(:out, Vv), (:ν, H.constant!(graph, 4.0)), (:S, H.constant!(graph, 2 * I2))]
    y = [H.data!(graph) for _ in 1:3]
    meanfield!(graph, InverseWishart, U_prior)
    meanfield!(graph, InverseWishart, V_prior)
    M = [0.5 0.0; 0.0 0.5]
    for i in 1:3
        meanfield!(graph, MatrixNormal, [(:out, y[i]), (:M, H.constant!(graph, M)), (:U, U), (:V, Vv)])
    end

    Y = [[1.0 0.5; 0.2 1.1], [0.8 0.3; -0.1 0.9], [0.2 -0.4; 0.6 0.3]]
    result = H.run(
        graph; data = [y => Y], iterations = 30, posteriors = [:U => U, :V => Vv],
        initial_marginals = [U => InverseWishart(5.0, I2), Vv => InverseWishart(4.0, 2 * I2)],
    )

    R = [Yi - M for Yi in Y]
    ΨU, ΨV = let
        ΨU, ΨV = I2, 2 * I2
        for _ in 1:500
            EVinv = 10.0 * inv(ΨV)
            ΨU = I2 + sum(Ri * EVinv * Ri' for Ri in R)
            EUinv = 11.0 * inv(ΨU)
            ΨV = 2 * I2 + sum(Ri' * EUinv * Ri for Ri in R)
        end
        (ΨU, ΨV)
    end
    q_U, q_V = result.posteriors["U"], result.posteriors["V"]
    @test q_U isa InverseWishart && q_V isa InverseWishart
    @test (q_U.df, q_V.df) == (11.0, 10.0)
    @test Matrix(q_U.Ψ) ≈ ΨU atol = 1.0e-10
    @test Matrix(q_V.Ψ) ≈ ΨV atol = 1.0e-10
    @test V.nonincreasing(result.free_energy)
    @test V.settled(result.free_energy)
end

@testitem "engine:scale-precision normal: coordinate ascent" tags = [:engine] setup = [EngineHarness, VariationalChecks] begin
    # μ ~ N(0, 10 I), γ ~ Gamma(2, 1), y[i] ~ N(μ, (γ I)⁻¹) under mean-field:
    # q(μ) = N with precision I/10 + n E[γ] I and weighted mean E[γ] Σy, and
    # q(γ) = Gamma(2 + n d/2, 1 + Σ E‖y - μ‖²/2).
    using ExponentialFamily, Distributions, LinearAlgebra, StandardMessagePassingRules
    H, V = EngineHarness, VariationalChecks
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

    Y = [[1.0, 0.5], [0.3, 1.2], [0.8, 0.9]]
    result = H.run(
        graph; data = [y => Y], iterations = 20,
        posteriors = [:γ => γ, :μ => μ], initial_marginals = [γ => GammaShapeRate(2.0, 1.0)],
    )

    a, b, m, v = let
        a, b = 2.0, 1.0
        local m, v
        for _ in 1:200
            Eγ = a / b
            v = 1 / (0.1 + 3 * Eγ)
            m = v * Eγ * sum(Y)
            a, b = 2 + 3, 1 + sum(sum(abs2, Yi - m) + 2v for Yi in Y) / 2
        end
        (a, b, m, v)
    end
    @test mean(result.posteriors["μ"]) ≈ m atol = 1.0e-10
    @test cov(result.posteriors["μ"]) ≈ v * I(2) atol = 1.0e-10
    @test shape(result.posteriors["γ"]) ≈ a atol = 1.0e-10
    @test rate(result.posteriors["γ"]) ≈ b atol = 1.0e-10
    @test V.nonincreasing(result.free_energy)
    @test V.settled(result.free_energy)
end

@testitem "engine:scale-matrix-precision normal: coordinate ascent" tags = [:engine] setup = [EngineHarness, VariationalChecks] begin
    # μ ~ N(0, 10 I), γ ~ Gamma(2, 1), G ~ Wishart(3, I), y[i] ~ N(μ, (γ G)⁻¹) under mean-field.
    # With D = Σ E[(y - μ)(y - μ)ᵀ]: q(μ) has precision I/10 + n E[γ] E[G] and weighted mean
    # E[γ] E[G] Σy, q(γ) = Gamma(2 + n d/2, 1 + tr(E[G] D)/2), q(G) = Wishart(3 + n, (I + E[γ] D)⁻¹).
    using ExponentialFamily, Distributions, LinearAlgebra, StandardMessagePassingRules
    H, V = EngineHarness, VariationalChecks
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

    Y = [[1.0, 0.5], [0.3, 1.2], [0.8, 0.9]]
    result = H.run(
        graph; data = [y => Y], iterations = 30,
        posteriors = [:γ => γ, :μ => μ, :G => G], initial_marginals = [γ => GammaShapeRate(2.0, 1.0), G => Wishart(3.0, I2)],
    )

    a, b, ν, S, m, Σ = let
        a, b, ν, S = 2.0, 1.0, 3.0, I2
        local m, Σ
        for _ in 1:500
            Eγ, EG = a / b, ν * S
            Σ = inv(I2 / 10 + 3 * Eγ * EG)
            m = Σ * (Eγ * EG * sum(Y))
            D = sum((Yi - m) * (Yi - m)' + Σ for Yi in Y)
            a, b = 2 + 3, 1 + tr(EG * D) / 2
            ν, S = 3 + 3, inv(I2 + (a / b) * D)
        end
        (a, b, ν, S, m, Σ)
    end
    q_μ, q_γ, q_G = result.posteriors["μ"], result.posteriors["γ"], result.posteriors["G"]
    @test mean(q_μ) ≈ m atol = 1.0e-9
    @test cov(q_μ) ≈ Σ atol = 1.0e-9
    @test shape(q_γ) ≈ a atol = 1.0e-9
    @test rate(q_γ) ≈ b atol = 1.0e-9
    @test q_G isa Wishart && q_G.df == 6.0
    @test Matrix(q_G.S) ≈ S atol = 1.0e-9
    @test V.settled(result.free_energy)
end

@testitem "engine:autoregressive chain under mean-field: a fixed point of coordinate ascent" tags = [:engine] setup = [EngineHarness, VariationalChecks] begin
    # θ ~ N(0.5, 1), γ ~ Gamma(2, 1), x0 ~ N(0, 1), x[i] ~ N(θ x[i - 1], 1/γ), y[i] ~ N(x[i], 0.1),
    # three steps, every variable a factor of its own; the fixed point by coordinate ascent.
    using ExponentialFamily, Distributions, StandardMessagePassingRules, AutoregressiveMessagePassingRules
    H, V = EngineHarness, VariationalChecks
    node!(graph, fform, interfaces; kwargs...) = H.node!(graph, fform, interfaces; factorisation = H.meanfield_factorisation(interfaces), kwargs...)

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

    Y = [0.8, 0.5, 0.3]
    result = H.run(
        graph; data = y .=> Y, iterations = 40, posteriors = [:γ => γ, :θ => θ, :x => [x0; x]],
        initial_marginals = [θ => NormalMeanVariance(0.5, 1.0), γ => GammaShapeRate(2.0, 1.0), x0 => NormalMeanVariance(0.0, 1.0), (x .=> Ref(NormalMeanVariance(0.0, 1.0)))...],
    )

    # One sweep of the coordinate-ascent updates from the engine's posteriors must give them
    # back; x[0] = x0 is m[1], v[1].
    q_x, q_θ, q_γ = result.posteriors["x"], result.posteriors["θ"], result.posteriors["γ"]
    m, v = mean.(q_x), var.(q_x)
    mθ, vθ, Eγ = mean(q_θ), var(q_θ), shape(q_γ) / rate(q_γ)
    Eθ², Ex² = mθ^2 + vθ, m .^ 2 .+ v
    for t in 0:3
        λ = (t == 0 ? 1.0 : 10.0 + Eγ) + (t < 3 ? Eγ * Eθ² : 0.0)
        ξ = (t == 0 ? 0.0 : 10.0 * Y[t] + Eγ * mθ * m[t]) + (t < 3 ? Eγ * mθ * m[t + 2] : 0.0)
        @test precision(q_x[t + 1]) ≈ λ atol = 1.0e-9
        @test mean(q_x[t + 1]) ≈ ξ / λ atol = 1.0e-9
    end
    λθ = 1.0 + Eγ * sum(Ex²[1:3])
    @test precision(q_θ) ≈ λθ atol = 1.0e-9
    @test mθ ≈ (0.5 + Eγ * sum(m[t + 1] * m[t] for t in 1:3)) / λθ atol = 1.0e-9
    # q(γ) = Gamma(2 + 3/2, 1 + Σ E[(x[t] - θ x[t - 1])²]/2), and E[θ² x²] = E[θ²] E[x²]. The
    # mean-field rule towards γ leaves out Var[θ] Var[x[t - 1]] from it, so this does not hold.
    @test shape(q_γ) ≈ 2 + 3 / 2 atol = 1.0e-12
    @test_broken rate(q_γ) ≈ 1 + sum(Ex²[t + 1] - 2 * mθ * m[t + 1] * m[t] + Eθ² * Ex²[t] for t in 1:3) / 2 atol = 1.0e-9
    @test V.settled(result.free_energy)
end

@testmodule AutoregressiveReference begin
    # An AR(2) of three steps, x[t] = (s[t], s[t - 1]) with s[t] ~ N(θ ⋅ (s[t - 1], s[t - 2]), 1/γ),
    # x0 = (s[0], s[-1]) ~ N(0, I), y[t] ~ N(x[t], 0.1 I): the chain of states is the scalar
    # sequence s[-1:3], Gaussian given the expectations the transitions read from q(θ, γ).
    using LinearAlgebra
    const Y = [[0.8, 0.1], [0.5, 0.8], [0.3, 0.5]]

    # The joint of s[-1:3] (s[t] at index t + 2) given E[γ], E[γ θ] and E[γ θ θᵀ].
    function states(Eγ, Eγθ, Eγθθ)
        Λ, h = zeros(5, 5), zeros(5)
        Λ[1, 1] += 1.0
        Λ[2, 2] += 1.0
        for t in 1:3
            Λ[t + 2, t + 2] += 10.0
            Λ[t + 1, t + 1] += 10.0
            h[t + 2] += 10.0 * Y[t][1]
            h[t + 1] += 10.0 * Y[t][2]
            block = [t + 2, t + 1, t] # s[t], then u[t] = (s[t - 1], s[t - 2])
            Λ[block, block] .+= [Eγ -Eγθ'; -Eγθ Eγθθ]
        end
        Σ = inv(Λ)
        return Σ * h, Σ
    end

    # E[s[t]²], E[u[t] s[t]] and E[u[t] u[t]ᵀ] from the joint.
    function moments(m, Σ, t)
        S = Σ + m * m'
        u = [t + 1, t]
        return S[t + 2, t + 2], S[u, t + 2], S[u, u]
    end

    # (mean, covariance) of x[t] = (s[t], s[t - 1]), x[0] = x0 first.
    marginals(m, Σ) = [(m[[t + 2, t + 1]], Σ[[t + 2, t + 1], [t + 2, t + 1]]) for t in 0:3]
end

@testitem "engine:autoregressive chain under a structured factorisation: coordinate ascent" tags = [:engine] setup = [EngineHarness, VariationalChecks, AutoregressiveReference] begin
    # Under q(x0, x) q(θ) q(γ): given q(θ) and q(γ) the chain is exact; q(θ) has precision
    # I + E[γ] Σ E[u uᵀ] and weighted mean (0.5, 0) + E[γ] Σ E[u s], and q(γ) = Gamma(2 + 3/2,
    # 1 + Σ(E[s²] - 2 E[θ] ⋅ E[u s] + tr(E[θ θᵀ] E[u uᵀ]))/2). ARsafe stands in for the noiseless
    # second entry with a precision of 1e12, which the reference takes as exact; inverting it
    # costs the engine about six digits (measured, 1.4e-6; under ARunsafe the two agree to 1e-8).
    using ExponentialFamily, Distributions, LinearAlgebra, StandardMessagePassingRules, AutoregressiveMessagePassingRules
    H, V, R = EngineHarness, VariationalChecks, AutoregressiveReference
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

    result = H.run(
        graph; data = y .=> R.Y, iterations = 60, posteriors = [:γ => γ, :θ => θ, :x => [x0; x]],
        initial_marginals = [θ => MvNormalMeanCovariance([0.5, 0.0], I2), γ => GammaShapeRate(2.0, 1.0)],
    )

    function coordinate_ascent(; sweeps = 2000)
        mθ, Vθ, a, b = [0.5, 0.0], Matrix(I2), 2.0, 1.0
        local m, Σ
        for _ in 1:sweeps
            Eγ, Eθθ = a / b, Vθ + mθ * mθ'
            m, Σ = R.states(Eγ, Eγ * mθ, Eγ * Eθθ)
            stats = [R.moments(m, Σ, t) for t in 1:3]
            Λθ = I2 + Eγ * sum(S[3] for S in stats)
            Vθ = inv(Λθ)
            mθ = Vθ * ([0.5, 0.0] + Eγ * sum(S[2] for S in stats))
            Eθθ = Vθ + mθ * mθ'
            a, b = 2.0 + 3 / 2, 1.0 + sum(S[1] - 2 * dot(mθ, S[2]) + tr(Eθθ * S[3]) for S in stats) / 2
        end
        return (; mθ, Vθ, a, b, x = R.marginals(m, Σ))
    end
    reference = coordinate_ascent()
    @test mean(result.posteriors["θ"]) ≈ reference.mθ atol = 1.0e-5
    @test cov(result.posteriors["θ"]) ≈ reference.Vθ atol = 1.0e-5
    @test shape(result.posteriors["γ"]) ≈ reference.a atol = 1.0e-12
    @test rate(result.posteriors["γ"]) ≈ reference.b atol = 1.0e-5
    for t in 1:4
        @test mean(result.posteriors["x"][t]) ≈ reference.x[t][1] atol = 1.0e-5
        @test cov(result.posteriors["x"][t]) ≈ reference.x[t][2] atol = 1.0e-5
    end
    @test V.settled(result.free_energy)
end

@testitem "engine:conjugate autoregressive chain under a structured factorisation: coordinate ascent" tags = [:engine] setup = [EngineHarness, VariationalChecks, AutoregressiveReference] begin
    # The AR(2) above with w = (θ, γ) ~ MvNormalGamma((0.5, 0), I, 2, 1), θ | γ ~ N(μ, (γ Λ)⁻¹),
    # under q(x0, x) q(w): q(w) is its conjugate update, Λₙ = Λ + Σ E[u uᵀ],
    # μₙ = Λₙ⁻¹ (Λ μ + Σ E[u s]), αₙ = α + 3/2, βₙ = β + (Σ E[s²] + μᵀΛμ - μₙᵀΛₙμₙ)/2, and the chain
    # reads E[γ] = α/β, E[γ θ] = E[γ] μₙ and E[γ θ θᵀ] = Λₙ⁻¹ + E[γ] μₙ μₙᵀ. As above, ARsafe's
    # precision of 1e12 costs about six digits.
    using ExponentialFamily, Distributions, LinearAlgebra, StandardMessagePassingRules, AutoregressiveMessagePassingRules
    H, V, R = EngineHarness, VariationalChecks, AutoregressiveReference
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

    result = H.run(
        graph; data = y .=> R.Y, iterations = 60, posteriors = [:w => w, :x => [x0; x]],
        initial_marginals = [w => MvNormalGamma([0.5, 0.0], I2, 2.0, 1.0)],
    )

    function coordinate_ascent(; sweeps = 2000)
        μ0, Λ0, α0, β0 = [0.5, 0.0], Matrix(I2), 2.0, 1.0
        μn, Λn, αn, βn = μ0, Λ0, α0, β0
        local m, Σ
        for _ in 1:sweeps
            Eγ = αn / βn
            m, Σ = R.states(Eγ, Eγ * μn, inv(Λn) + Eγ * μn * μn')
            stats = [R.moments(m, Σ, t) for t in 1:3]
            Λn = Λ0 + sum(S[3] for S in stats)
            μn = Λn \ (Λ0 * μ0 + sum(S[2] for S in stats))
            αn = α0 + 3 / 2
            βn = β0 + (sum(S[1] for S in stats) + dot(μ0, Λ0 * μ0) - dot(μn, Λn * μn)) / 2
        end
        return (; μn, Λn, αn, βn, x = R.marginals(m, Σ))
    end
    reference = coordinate_ascent()
    q_w = result.posteriors["w"]
    @test q_w isa MvNormalGamma
    @test q_w.μ ≈ reference.μn atol = 1.0e-5
    @test q_w.Λ ≈ reference.Λn atol = 1.0e-5
    @test q_w.α ≈ reference.αn atol = 1.0e-12
    @test q_w.β ≈ reference.βn atol = 1.0e-5
    for t in 1:4
        @test mean(result.posteriors["x"][t]) ≈ reference.x[t][1] atol = 1.0e-5
        @test cov(result.posteriors["x"][t]) ≈ reference.x[t][2] atol = 1.0e-5
    end
    @test V.settled(result.free_energy)
end

@testmodule ContinuousTransitionModel begin
    # y[i] ~ N(x[i], 0.1 I) observed, x[i] ~ ContinuousTransition(x[i - 1], a, W) with
    # A = reshape(a, 2, 2): x[i] ~ N(A x[i - 1], W⁻¹), a ~ N(vec(I), I), W ~ Wishart(4, I).
    using ExponentialFamily, Distributions, StandardMessagePassingRules, ContinuousTransitionMessagePassingRules
    import ..EngineHarness as H

    const I2, I4 = [1.0 0.0; 0.0 1.0], [1.0 0.0 0.0 0.0; 0.0 1.0 0.0 0.0; 0.0 0.0 1.0 0.0; 0.0 0.0 0.0 1.0]
    const Y = [[1.0, 0.2], [0.8, 0.5], [0.5, 0.7]]

    function run(; factorisation, initial_marginals, iterations)
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
            graph; data = y .=> Y, iterations, posteriors = [:a => a, :W => W, :x => [x0; x]],
            initial_marginals = initial_marginals(variables),
        )
    end
end

@testitem "engine:continuous transition under mean-field: coordinate ascent" tags = [:engine] setup = [EngineHarness, VariationalChecks, ContinuousTransitionModel] begin
    # Every variable a factor of its own. With A x = (xᵀ ⊗ I) a, and E[A]ᵢₖ = E[a][(k - 1) 2 + i]:
    # q(a) has precision I + Σ E[x xᵀ] ⊗ E[W] and weighted mean vec(I) + Σ vec(E[W] E[x[i]] E[x[i - 1]]ᵀ);
    # q(W) = Wishart(4 + 3, (I + Σ E[(x[i] - A x[i - 1])(x[i] - A x[i - 1])ᵀ])⁻¹); and q(x[i]) reads
    # E[W], E[A] and E[Aᵀ W A] from its two transitions.
    using ExponentialFamily, Distributions, LinearAlgebra
    M, V = ContinuousTransitionModel, VariationalChecks
    initial(v) = [
        v.a => MvNormalMeanCovariance([1.0, 0.0, 0.0, 1.0], M.I4), v.W => Wishart(4, M.I2), v.x0 => MvNormalMeanCovariance([0.0, 0.0], M.I2),
        (v.x .=> Ref(MvNormalMeanCovariance([0.0, 0.0], M.I2)))...,
    ]
    result = M.run(; factorisation = ((:y,), (:x,), (:a,), (:W,)), initial_marginals = initial, iterations = 60)

    function coordinate_ascent(; sweeps = 2000)
        I2 = M.I2
        ma, Va = [1.0, 0.0, 0.0, 1.0], Matrix(M.I4)
        ν, S = 4.0, I2
        m, P = [zeros(2) for _ in 0:3], [I2 for _ in 0:3] # x[0:3] as m[1:4], covariances P
        EA() = reshape(ma, 2, 2)
        # E[A X Bᵀ]-like moments: E[Aᵢₖ Aⱼₗ] = E[A]ᵢₖ E[A]ⱼₗ + Va[(k - 1) 2 + i, (l - 1) 2 + j]
        EAik_Ajl(i, k, j, l) = EA()[i, k] * EA()[j, l] + Va[(k - 1) * 2 + i, (l - 1) * 2 + j]
        EAXAt(X) = [sum(X[k, l] * EAik_Ajl(i, k, j, l) for k in 1:2, l in 1:2) for i in 1:2, j in 1:2]
        EAtWA(W) = [sum(W[i, j] * EAik_Ajl(i, k, j, l) for i in 1:2, j in 1:2) for k in 1:2, l in 1:2]
        for _ in 1:sweeps
            EW = ν * S
            for t in 0:3
                λ = t == 0 ? Matrix(I2) : 10.0 * I2 + EW
                ξ = t == 0 ? zeros(2) : 10.0 * M.Y[t] + EW * EA() * m[t]
                if t < 3
                    λ += EAtWA(EW)
                    ξ += EA()' * EW * m[t + 2]
                end
                P[t + 1] = inv(λ)
                m[t + 1] = P[t + 1] * ξ
            end
            X(t) = P[t + 1] + m[t + 1] * m[t + 1]'
            Λa = Matrix(M.I4) + sum(kron(X(t - 1), EW) for t in 1:3)
            ξa = [1.0, 0.0, 0.0, 1.0] + sum(vec(EW * m[t + 1] * m[t]') for t in 1:3)
            Va = inv(Λa)
            ma = Va * ξa
            D = sum(X(t) - m[t + 1] * (EA() * m[t])' - (EA() * m[t]) * m[t + 1]' + EAXAt(X(t - 1)) for t in 1:3)
            ν, S = 4.0 + 3, inv(I2 + D)
        end
        return (; ma, Va, ν, S, m, P)
    end
    reference = coordinate_ascent()
    @test mean(result.posteriors["a"]) ≈ reference.ma atol = 1.0e-8
    @test cov(result.posteriors["a"]) ≈ reference.Va atol = 1.0e-8
    @test result.posteriors["W"].df ≈ reference.ν atol = 1.0e-12
    @test Matrix(result.posteriors["W"].S) ≈ reference.S atol = 1.0e-8
    for t in 1:4
        @test mean(result.posteriors["x"][t]) ≈ reference.m[t] atol = 1.0e-8
        @test cov(result.posteriors["x"][t]) ≈ reference.P[t] atol = 1.0e-8
    end
    @test V.settled(result.free_energy)
end

@testitem "engine:continuous transition under a structured factorisation: coordinate ascent" tags = [:engine] setup = [EngineHarness, VariationalChecks, ContinuousTransitionModel] begin
    # Under q(x0, x) q(a) q(W): given q(a) and q(W) the chain is exact, a joint Gaussian of
    # x[0:3] whose transitions read E[W], E[W] E[A] and E[Aᵀ W A]; q(a) and q(W) are the updates of
    # the mean-field case, read from the joint's moments, E[x[i] x[i - 1]ᵀ] included. The rule
    # towards `y` from the message on `x`, N(E[A] m, E[A] V E[A]ᵀ + E[W]⁻¹), leaves out the
    # uncertainty of `a` that E[Aᵀ W A] carries, so the engine settles elsewhere, by about 1e-2:
    # those comparisons are broken.
    using ExponentialFamily, Distributions, LinearAlgebra
    M, V = ContinuousTransitionModel, VariationalChecks
    initial(v) = [v.a => MvNormalMeanCovariance([1.0, 0.0, 0.0, 1.0], M.I4), v.W => Wishart(4, M.I2)]
    result = M.run(; factorisation = ((:y, :x), (:a,), (:W,)), initial_marginals = initial, iterations = 60)

    function coordinate_ascent(; sweeps = 2000)
        I2 = M.I2
        ma, Va = [1.0, 0.0, 0.0, 1.0], Matrix(M.I4)
        ν, S = 4.0, I2
        block(t) = (2t + 1):(2t + 2) # x[t] in the joint of x[0:3]
        local m, Σ
        for _ in 1:sweeps
            EW, EA = ν * S, reshape(ma, 2, 2)
            EAA(i, k, j, l) = EA[i, k] * EA[j, l] + Va[(k - 1) * 2 + i, (l - 1) * 2 + j]
            EAtWA = [sum(EW[i, j] * EAA(i, k, j, l) for i in 1:2, j in 1:2) for k in 1:2, l in 1:2]
            Λ, h = zeros(8, 8), zeros(8)
            Λ[block(0), block(0)] .+= I2
            for t in 1:3
                Λ[block(t), block(t)] .+= 10.0 * I2 + EW
                Λ[block(t - 1), block(t - 1)] .+= EAtWA
                Λ[block(t), block(t - 1)] .-= EW * EA
                Λ[block(t - 1), block(t)] .-= EA' * EW
                h[block(t)] .+= 10.0 * M.Y[t]
            end
            Σ = inv(Λ)
            m = Σ * h
            moment(t, u) = Σ[block(t), block(u)] + m[block(t)] * m[block(u)]'
            Λa = Matrix(M.I4) + sum(kron(moment(t - 1, t - 1), EW) for t in 1:3)
            Va = inv(Λa)
            ma = Va * ([1.0, 0.0, 0.0, 1.0] + sum(vec(EW * moment(t, t - 1)) for t in 1:3))
            EA = reshape(ma, 2, 2)
            EAXAt(X) = [sum(X[k, l] * (EA[i, k] * EA[j, l] + Va[(k - 1) * 2 + i, (l - 1) * 2 + j]) for k in 1:2, l in 1:2) for i in 1:2, j in 1:2]
            D = sum(moment(t, t) - moment(t, t - 1) * EA' - EA * moment(t - 1, t) + EAXAt(moment(t - 1, t - 1)) for t in 1:3)
            ν, S = 4.0 + 3, inv(I2 + D)
        end
        return (; ma, Va, ν, S, x = [(m[block(t)], Σ[block(t), block(t)]) for t in 0:3])
    end
    reference = coordinate_ascent()
    @test_broken mean(result.posteriors["a"]) ≈ reference.ma atol = 1.0e-8
    @test_broken cov(result.posteriors["a"]) ≈ reference.Va atol = 1.0e-8
    @test result.posteriors["W"].df ≈ reference.ν atol = 1.0e-12
    @test_broken Matrix(result.posteriors["W"].S) ≈ reference.S atol = 1.0e-8
    @test_broken all(t -> isapprox(mean(result.posteriors["x"][t]), reference.x[t][1]; atol = 1.0e-8), 1:4)
    @test_broken all(t -> isapprox(cov(result.posteriors["x"][t]), reference.x[t][2]; atol = 1.0e-8), 1:4)
    @test isposdef(cov(result.posteriors["a"])) && isposdef(Matrix(result.posteriors["W"].S))
    @test all(q -> isposdef(cov(q)), result.posteriors["x"])
    @test V.settled(result.free_energy)
end

@testmodule PolyaGamma begin
    # The mean of PG(n, c), n/(2c) tanh(c/2), n/4 at c = 0.
    mean_pg(n, c) = abs(c) < 1.0e-8 ? n / 4 : n / (2c) * tanh(c / 2)
    softplus(x) = x > 0 ? x + log1p(exp(-x)) : log1p(exp(x))
    # E[softplus(ψ)] for ψ ~ N(m, v), by the trapezoidal rule over ±12 standard deviations.
    function expected_softplus(m, v)
        s = sqrt(v)
        grid = range(m - 12s, m + 12s; length = 20_001)
        density = exp.(-(grid .- m) .^ 2 ./ (2v)) ./ sqrt(2π * v)
        return sum(density .* softplus.(grid)) * step(grid)
    end
end

@testitem "engine:binomial regression through pólya-gamma augmentation: its fixed point" tags = [:engine] setup = [EngineHarness, VariationalChecks, PolyaGamma] begin
    # β ~ N(0, I), y[i] ~ Binomial(n[i], σ(X[i] ⋅ β)). Each node's message towards β is the
    # augmented likelihood N(ξ = (y - n/2) X, Λ = ω X Xᵀ), with ω the Pólya-Gamma mean at X ⋅ μ,
    # μ the mean of the message on its own edge (the prior times the other nodes' messages).
    # Iterated here to its fixed point. On this tree the free energy is the variational one,
    # E_q[-log p(β)] + Σ E_q[-log p(y | β)] - H[q].
    using ExponentialFamily, Distributions, LinearAlgebra, SpecialFunctions, StandardMessagePassingRules, PolyaMessagePassingRules
    H, V, P = EngineHarness, VariationalChecks, PolyaGamma
    prior = MvNormalWeightedMeanPrecision([0.0, 0.0], [1.0 0.0; 0.0 1.0])

    graph = H.Graph()
    β = H.random!(graph)
    H.node!(graph, MvNormalWeightedMeanPrecision, [(:out, β), (:ξ, H.constant!(graph, [0.0, 0.0])), (:Λ, H.constant!(graph, [1.0 0.0; 0.0 1.0]))])
    ys, Xs, ns = ([H.data!(graph) for _ in 1:3] for _ in 1:3)
    for i in 1:3
        H.node!(graph, BinomialPolya, [(:y, ys[i]), (:x, Xs[i]), (:n, ns[i]), (:β, β)])
    end

    Y, X, N = [3.0, 1.0, 4.0], [[1.0, 0.5], [1.0, -0.3], [1.0, 1.2]], [5.0, 4.0, 5.0]
    data = [(ys .=> Y)..., (Xs .=> X)..., (ns .=> N)...]
    result = H.run(graph; data, iterations = 30, posteriors = [:β => β], initial_messages = [β => prior])

    function fixed_point(; sweeps = 200)
        ξs = [(Y[i] - N[i] / 2) * X[i] for i in 1:3]
        ωs = zeros(3)
        for _ in 1:sweeps, i in 1:3
            others = [j for j in 1:3 if j != i]
            Λc = I + sum(ωs[j] * X[j] * X[j]' for j in others)
            μc = Λc \ sum(ξs[others])
            ωs[i] = P.mean_pg(N[i], dot(X[i], μc))
        end
        Λ = I + sum(ωs[i] * X[i] * X[i]' for i in 1:3)
        return Λ \ sum(ξs), inv(Λ)
    end
    m, Σ = fixed_point()
    q = result.posteriors["β"]
    @test mean(q) ≈ m atol = 1.0e-9
    @test cov(q) ≈ Σ atol = 1.0e-9

    function likelihood_energy(i)
        ψm, ψv = dot(X[i], m), dot(X[i], Σ * X[i])
        logbinomial = loggamma(N[i] + 1) - loggamma(Y[i] + 1) - loggamma(N[i] - Y[i] + 1)
        return -logbinomial - Y[i] * ψm + N[i] * P.expected_softplus(ψm, ψv)
    end
    energy = log(2π) + (tr(Σ) + dot(m, m)) / 2 + sum(likelihood_energy, 1:3)
    @test last(result.free_energy) ≈ energy - entropy(MvNormal(m, Σ)) atol = 1.0e-8
    @test V.settled(result.free_energy)
end

@testitem "engine:multinomial regression through pólya-gamma augmentation: its fixed point" tags = [:engine] setup = [EngineHarness, VariationalChecks, PolyaGamma] begin
    # ψ ~ N(0, I), x[i] ~ Multinomial(10, stick-breaking(ψ)) observed: break k is a binomial of
    # x[k] out of n[k] = 10 - Σ_{j<k} x[j] with logit ψ[k], augmented as in the binomial case,
    # each break at the mean of the message on the node's own edge. The free energy is again the
    # variational one.
    using ExponentialFamily, Distributions, LinearAlgebra, SpecialFunctions, StandardMessagePassingRules, PolyaMessagePassingRules
    H, V, P = EngineHarness, VariationalChecks, PolyaGamma
    prior = MvNormalWeightedMeanPrecision([0.0, 0.0], [1.0 0.0; 0.0 1.0])

    graph = H.Graph()
    ψ = H.random!(graph)
    H.node!(graph, MvNormalWeightedMeanPrecision, [(:out, ψ), (:ξ, H.constant!(graph, [0.0, 0.0])), (:Λ, H.constant!(graph, [1.0 0.0; 0.0 1.0]))])
    ys, Nc = [H.data!(graph) for _ in 1:3], H.constant!(graph, 10)
    for i in 1:3
        H.node!(graph, MultinomialPolya, [(:x, ys[i]), (:N, Nc), (:ψ, ψ)])
    end

    counts = [[3, 2, 5], [1, 4, 5], [2, 2, 6]]
    result = H.run(graph; data = ys .=> counts, iterations = 30, posteriors = [:ψ => ψ], initial_messages = [ψ => prior])

    n = [[10, 10 - c[1]] for c in counts] # n[i][k]
    η = [[c[k] - n[i][k] / 2 for k in 1:2] for (i, c) in enumerate(counts)]
    function fixed_point(; sweeps = 200)
        ω = [zeros(2) for _ in 1:3]
        for _ in 1:sweeps, i in 1:3
            others = [j for j in 1:3 if j != i]
            λc = 1 .+ sum(ω[others])
            μc = sum(η[others]) ./ λc
            ω[i] = [P.mean_pg(n[i][k], μc[k]) for k in 1:2]
        end
        λ = 1 .+ sum(ω)
        return sum(η) ./ λ, 1 ./ λ
    end
    m, v = fixed_point()
    q = result.posteriors["ψ"]
    @test mean(q) ≈ m atol = 1.0e-9
    @test cov(q) ≈ Diagonal(v) atol = 1.0e-9

    function break_energy(i, k)
        c = counts[i]
        logbinomial = loggamma(n[i][k] + 1) - loggamma(c[k] + 1) - loggamma(n[i][k] - c[k] + 1)
        return -logbinomial - c[k] * m[k] + n[i][k] * P.expected_softplus(m[k], v[k])
    end
    energy = log(2π) + (sum(v) + dot(m, m)) / 2 + sum(break_energy(i, k) for i in 1:3, k in 1:2)
    @test last(result.free_energy) ≈ energy - entropy(MvNormal(m, Diagonal(v))) atol = 1.0e-8
    @test V.settled(result.free_energy)
end

@testitem "engine:hidden markov model with learned tensors: a fixed point of coordinate ascent" tags = [:engine] setup = [EngineHarness, VariationalChecks] begin
    # s0 ~ Cat(1/3), s[t] ~ A[:, s[t - 1]], x[t] ~ B[:, s[t]] observed, A and B column-wise
    # Dirichlet, under q(s0, s) q(A) q(B). Given q(A) and q(B), q(s) is the chain with the
    # tensors exp(E[log A]) and exp(E[log B]), exact by forward-backward; then q(A) and q(B) add
    # the chain's pairwise and single marginals to their priors. One sweep from the engine's
    # posteriors must give them back. Belief propagation along the chain mixes with the tensors'
    # updates, so the free energy need not decrease along the way.
    using ExponentialFamily, Distributions, SpecialFunctions, StandardMessagePassingRules, DiscreteTransitionMessagePassingRules
    H, V = EngineHarness, VariationalChecks
    I3 = [1.1 0.1 0.1; 0.1 1.1 0.1; 0.1 0.1 1.1]
    observed = [[1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0]]
    prior_A, prior_B = ones(3, 3), [10.0 1.0 1.0; 1.0 10.0 1.0; 1.0 1.0 10.0]

    graph = H.Graph()
    A, B, s0 = H.random!(graph), H.random!(graph), H.random!(graph)
    H.node!(graph, DirichletCollection, [(:out, A), (:a, H.constant!(graph, prior_A))])
    H.node!(graph, DirichletCollection, [(:out, B), (:a, H.constant!(graph, prior_B))])
    H.node!(graph, Categorical, [(:out, s0), (:p, H.constant!(graph, fill(1.0 / 3.0, 3)))])
    s, x = [H.random!(graph) for _ in observed], [H.data!(graph) for _ in observed]
    for t in eachindex(observed)
        H.node!(graph, DiscreteTransition, [(:out, s[t]), (:in, t == 1 ? s0 : s[t - 1]), (:a, A)]; factorisation = ((:out, :in), (:a,)))
        H.node!(graph, DiscreteTransition, [(:out, x[t]), (:in, s[t]), (:a, B)]; factorisation = ((:out,), (:in,), (:a,)))
    end

    result = H.run(
        graph; data = x .=> observed, iterations = 60, posteriors = [:A => A, :B => B, :s => [s0; s]],
        initial_marginals = [A => DirichletCollection(I3), B => DirichletCollection(I3)],
    )

    expected_log(α) = digamma.(α) .- digamma.(sum(α; dims = 1))
    Ã, B̃ = exp.(expected_log(result.posteriors["A"].α)), exp.(expected_log(result.posteriors["B"].α))
    T = length(observed)
    evidence = [B̃' * o for o in observed] # the emission factor of s[t]
    # Forward-backward over s[0:T]: α[t + 1] and β[t + 1] belong to s[t].
    forward = Vector{Vector{Float64}}(undef, T + 1)
    backward = Vector{Vector{Float64}}(undef, T + 1)
    forward[1] = fill(1 / 3, 3)
    for t in 1:T
        f = (Ã * forward[t]) .* evidence[t]
        forward[t + 1] = f ./ sum(f)
    end
    backward[T + 1] = ones(3)
    for t in T:-1:1
        b = Ã' * (evidence[t] .* backward[t + 1])
        backward[t] = b ./ sum(b)
    end
    marginals = [forward[t] .* backward[t] ./ sum(forward[t] .* backward[t]) for t in 1:(T + 1)]
    pairwise = map(1:T) do t # ξ[i, j] = q(s[t] = i, s[t - 1] = j)
        ξ = (evidence[t] .* backward[t + 1]) .* Ã .* forward[t]'
        ξ ./ sum(ξ)
    end

    for t in 1:(T + 1)
        @test probs(result.posteriors["s"][t]) ≈ marginals[t] atol = 1.0e-8
    end
    @test result.posteriors["A"].α ≈ prior_A .+ sum(pairwise) atol = 1.0e-8
    @test result.posteriors["B"].α ≈ prior_B .+ sum(observed[t] * marginals[t + 1]' for t in 1:T) atol = 1.0e-8
    @test V.settled(result.free_energy)
end

@testitem "engine:a joint over part of a group: a fixed point and its free energy" tags = [:engine] setup = [EngineHarness, VariationalChecks] begin
    # z ~ A[:, x, t1, t2], y ~ [0.9 0.2; 0.1 0.8][:, z] observed at 2, with x, t1, t2 categorical
    # and A a Dirichlet collection, under q(z, t1) q(x) q(t2) q(A): a joint of `out` and only the
    # first member of the group T. From the engine's posteriors, q(z, t1), q(x), q(t2) and q(A)
    # are recomputed by their coordinate-ascent updates, and the free energy by hand. The first
    # iterations' free energy is not monotone: the joint's first marginal reads marginals that are
    # not yet each other's updates.
    using ExponentialFamily, Distributions, SpecialFunctions, StandardMessagePassingRules, DiscreteTransitionMessagePassingRules
    H, V = EngineHarness, VariationalChecks

    graph = H.Graph()
    A = H.random!(graph)
    H.node!(graph, DirichletCollection, [(:out, A), (:a, H.constant!(graph, ones(2, 2, 3, 2)))])
    x, t1, t2 = H.random!(graph), H.random!(graph), H.random!(graph)
    px, pt1, pt2 = [0.4, 0.6], [0.2, 0.3, 0.5], [0.5, 0.5]
    H.node!(graph, Categorical, [(:out, x), (:p, H.constant!(graph, px))])
    H.node!(graph, Categorical, [(:out, t1), (:p, H.constant!(graph, pt1))])
    H.node!(graph, Categorical, [(:out, t2), (:p, H.constant!(graph, pt2))])
    z, y = H.random!(graph), H.data!(graph)
    H.node!(
        graph, DiscreteTransition, [(:out, z), (:in, x), (:a, A), ((:T, 1), t1), ((:T, 2), t2)];
        factorisation = ((:out, (:T, 1)), (:in,), (:a,), ((:T, 2),)),
    )
    Bm = [0.9 0.2; 0.1 0.8]
    H.node!(graph, DiscreteTransition, [(:out, y), (:in, z), (:a, H.constant!(graph, Bm))])

    result = H.run(
        graph; data = [y => [0.0, 1.0]], iterations = 30, posteriors = [:t2 => t2, :A => A, :z => z, :t1 => t1, :x => x],
        initial_marginals = [A => DirichletCollection(ones(2, 2, 3, 2)), x => Categorical([0.5, 0.5]), t2 => Categorical([0.5, 0.5])],
    )
    α = result.posteriors["A"].α
    ElogA = digamma.(α) .- digamma.(sum(α; dims = 1))
    qx, qt2 = probs(result.posteriors["x"]), probs(result.posteriors["t2"])
    normalise(w) = w ./ sum(w)

    # q(z, t1) ∝ p(t1) p(y | z) exp(E[log A[z, x, t1, t2]]), the expectation over q(x) q(t2).
    logjoint = [log(pt1[k]) + log(Bm[2, j]) + sum(qx[i] * qt2[l] * ElogA[j, i, k, l] for i in 1:2, l in 1:2) for j in 1:2, k in 1:3]
    qzt1 = normalise(exp.(logjoint .- maximum(logjoint)))
    @test probs(result.posteriors["z"]) ≈ vec(sum(qzt1; dims = 2)) atol = 1.0e-9
    @test probs(result.posteriors["t1"]) ≈ vec(sum(qzt1; dims = 1)) atol = 1.0e-9
    logx = [log(px[i]) + sum(qzt1[j, k] * qt2[l] * ElogA[j, i, k, l] for j in 1:2, k in 1:3, l in 1:2) for i in 1:2]
    @test qx ≈ normalise(exp.(logx .- maximum(logx))) atol = 1.0e-9
    logt2 = [log(pt2[l]) + sum(qzt1[j, k] * qx[i] * ElogA[j, i, k, l] for j in 1:2, i in 1:2, k in 1:3) for l in 1:2]
    @test qt2 ≈ normalise(exp.(logt2 .- maximum(logt2))) atol = 1.0e-9
    @test α ≈ [1 + qzt1[j, k] * qx[i] * qt2[l] for j in 1:2, i in 1:2, k in 1:3, l in 1:2] atol = 1.0e-9

    # F = E[-log p(A)] - H[q(A)] + E[-log p(x)] - H[q(x)] + E[-log p(t1)] + E[-log p(t2)] - H[q(t2)]
    #   + E[-log A[z, x, t1, t2]] - H[q(z, t1)] + E[-log p(y | z)]; the prior on A is flat, so its
    # energy is 0; t1's and z's own entropies cancel against their degrees.
    qz, qt1 = vec(sum(qzt1; dims = 2)), vec(sum(qzt1; dims = 1))
    entropy_A = sum(entropy(Dirichlet(α[:, i, k, l])) for i in 1:2, k in 1:3, l in 1:2)
    energy = -sum(qx .* log.(px)) - sum(qt1 .* log.(pt1)) - sum(qt2 .* log.(pt2)) - sum(qz .* log.(Bm[2, :]))
    energy -= sum(qzt1[j, k] * qx[i] * qt2[l] * ElogA[j, i, k, l] for j in 1:2, i in 1:2, k in 1:3, l in 1:2)
    entropies = entropy_A + entropy(Categorical(qx)) + entropy(Categorical(qt2)) - sum(p -> p * log(p), qzt1)
    @test last(result.free_energy) ≈ energy - entropies atol = 1.0e-9
    # It bounds -log p(y) = -log(Σ_z p(y | z) E[A[z, ⋅]]) = -log((0.1 + 0.8)/2) from above.
    @test last(result.free_energy) >= -log(0.45)
    @test V.settled(result.free_energy)
end

@testitem "engine:gaussian controlled variance under mean-field: a fixed point" tags = [:engine] setup = [EngineHarness, VariationalChecks] begin
    # x ~ N(0.5, 1), z ~ N(0, 1), y ~ N(x, exp(κ z + ω)) with κ = 1, ω = -0.5, o ~ N(y, 0.1)
    # observed at 2, every variable a factor of its own. With W = E[exp(-(κ z + ω))]:
    # q(y) has precision 10 + W and weighted mean 20 + W E[x]; q(x) precision 1 + W and weighted
    # mean 0.5 + W E[y]; q(z) is the normal of the moments of N(z; 0, 1) exp(-(κ z + ω)/2 -
    # E[(y - x)²] exp(-(κ z + ω))/2), which the engine's product computes by 20-point Gauss–Hermite
    # cubature, here by the trapezoidal rule: the cubature is accurate to about 5e-6 on this
    # integrand, hence the tolerance. The projection is no exact coordinate step, so no
    # monotonicity.
    using ExponentialFamily, Distributions, StandardMessagePassingRules, GCVMessagePassingRules
    H, V = EngineHarness, VariationalChecks
    node!(graph, fform, interfaces) = H.node!(graph, fform, interfaces; factorisation = H.meanfield_factorisation(interfaces))

    graph = H.Graph()
    x, y, z = H.random!(graph), H.random!(graph), H.random!(graph)
    o = H.data!(graph)
    node!(graph, NormalMeanVariance, [(:out, x), (:μ, H.constant!(graph, 0.5)), (:v, H.constant!(graph, 1.0))])
    node!(graph, NormalMeanVariance, [(:out, z), (:μ, H.constant!(graph, 0.0)), (:v, H.constant!(graph, 1.0))])
    node!(graph, GCV, [(:y, y), (:x, x), (:z, z), (:κ, H.constant!(graph, 1.0)), (:ω, H.constant!(graph, -0.5))])
    node!(graph, NormalMeanVariance, [(:out, o), (:μ, y), (:v, H.constant!(graph, 0.1))])

    result = H.run(
        graph; data = [o => 2.0], iterations = 40, posteriors = [:x => x, :y => y, :z => z],
        initial_marginals = [x => NormalMeanVariance(0.5, 1.0), y => NormalMeanVariance(2.0, 1.0), z => NormalMeanVariance(0.0, 1.0)],
    )
    q_x, q_y, q_z = result.posteriors["x"], result.posteriors["y"], result.posteriors["z"]
    κ, ω = 1.0, -0.5
    W = exp(-(κ * mean(q_z) + ω) + κ^2 * var(q_z) / 2)
    @test precision(q_y) ≈ 10 + W atol = 1.0e-9
    @test mean(q_y) * precision(q_y) ≈ 20 + W * mean(q_x) atol = 1.0e-9
    @test precision(q_x) ≈ 1 + W atol = 1.0e-9
    @test mean(q_x) * precision(q_x) ≈ 0.5 + W * mean(q_y) atol = 1.0e-9

    spread = (mean(q_y) - mean(q_x))^2 + var(q_y) + var(q_x)
    grid = range(-12.0, 12.0; length = 200_001)
    tilted = exp.(-grid .^ 2 ./ 2 .- (κ .* grid .+ ω) ./ 2 .- spread .* exp.(-(κ .* grid .+ ω)) ./ 2)
    Z = sum(tilted)
    m = sum(grid .* tilted) / Z
    @test mean(q_z) ≈ m atol = 2.0e-5
    @test var(q_z) ≈ sum((grid .- m) .^ 2 .* tilted) / Z atol = 2.0e-5
    @test V.settled(result.free_energy)
end
