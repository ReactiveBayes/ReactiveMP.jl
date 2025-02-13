@testitem "Basic out rule tests #1" begin
    using ExponentialFamily, ExponentialFamilyProjection, BayesBase

    @testset "f(x) = x, x ~ EF" begin
        meta = DeltaMeta(method = CVIProjection(), inverse = nothing)

        q_ins_m_out_incomings = [
            (FactorizedJoint((NormalMeanVariance(2, 2),)), NormalMeanVariance(0, 1)),
            (FactorizedJoint((Gamma(2, 1),)), Gamma(2, 2)),
            (FactorizedJoint((Beta(5, 4),)), Beta(5, 5)),
            (FactorizedJoint((Rayleigh(4),)), Rayleigh(5.2)),
            (FactorizedJoint((Geometric(0.3),)), Geometric(0.9)),
            (FactorizedJoint((LogNormal(0.2, 1.0),)), LogNormal(3.0, 2.1)),
            (FactorizedJoint((Exponential(0.3),)), Exponential(4.3))
        ]

        for (q_in, m_out_incoming) in q_ins_m_out_incomings
            q_in_component = first(components(q_in))
            q_out = q_in_component
            msg = @call_rule DeltaFn{identity}(:out, Marginalisation) (m_out = m_out_incoming, q_out = q_out, q_ins = q_in, meta = meta)

            prj = ProjectedTo(ExponentialFamily.exponential_family_typetag(q_out), size(q_out)...)

            q_out_projected = project_to(prj, (x) -> logpdf(msg, x) + logpdf(m_out_incoming, x); initialpoint = q_out)
            @test mean(q_out_projected) ≈ mean(q_out) atol = 5e-1
            @test var(q_out_projected) ≈ var(q_out) atol = 2.0
            @test mode(q_out_projected) ≈ mode(q_out) atol = 5e-1
            if typeof(q_out) <: Union{Exponential, Beta, Gamma}
                @test mean(log, q_out_projected) ≈ mean(log, q_out) atol = 5e-1
            end
        end
    end
end

# In this test we are trying to check that `DeltaFn` node can accept arbitrary (univariate) inputs 
# and compute an outbound (multivariate) message. 
# We use a simple node function f(x) = [x; y] and we test the following assumptions:
# - `mean(m_out) ≈ [ mean(m_x), mean(m_y) ]`
@testitem "Basic out rule tests #2" begin
    using ExponentialFamily, ExponentialFamilyProjection, BayesBase, LinearAlgebra

    meta = DeltaMeta(method = CVIProjection(outsamples = 1000), inverse = nothing)
    f(x, y) = [x; y]
    q_ins_m_out_incomings = [
        (FactorizedJoint((NormalMeanVariance(3.0, 1.0), MvNormalMeanCovariance([2.0, 5.2], 3diageye(2)))), MvNormalMeanCovariance([2.0, 3.4, -1.0], Diagonal([0.2, 0.01, 4.0]))),
        (
            FactorizedJoint((MvNormalMeanCovariance([0.3, 0.9], diageye(2)), MvNormalMeanCovariance([2.0, 5.2], 3diageye(2)))),
            MvNormalMeanCovariance([2.0, 3.4, -1.0, 5.0], Diagonal([0.2, 2.0, 0.01, 4.0]))
        ),
        (
            FactorizedJoint((MvNormalMeanCovariance([2.0, 3.0, 0.1, 0.9], diageye(4)), MvNormalMeanCovariance([3.4, 7.6], 0.1diageye(2)))),
            MvNormalMeanCovariance([2.0, 3.4, -1.0, 5.0, 3.0, -10.0], Diagonal([0.2, 2.0, 0.01, 4.0, 1.0, 0.5]))
        )
    ]
    for (q_in, m_out_incoming) in q_ins_m_out_incomings
        q_in_components    = components(q_in)
        mean_in_components = mapreduce(mean, vcat, q_in_components)
        cov_in_components  = Diagonal(mapreduce(var, vcat, q_in_components))
        q_out              = MvNormalMeanCovariance(mean_in_components, cov_in_components)
        msg                = @call_rule DeltaFn{f}(:out, Marginalisation) (m_out = m_out_incoming, q_out = q_out, q_ins = q_in, meta = meta)

        prj = ProjectedTo(ExponentialFamily.exponential_family_typetag(q_out), size(q_out)...)

        q_out_projected = project_to(prj, (x) -> logpdf(msg, x) + logpdf(m_out_incoming, x); initial_point = q_out)
        @test mean(q_out_projected) ≈ mean(q_out) rtol = 5e-1
        @test var(q_out_projected) ≈ var(q_out) rtol = 1.0
        @test mode(q_out_projected) ≈ mode(q_out) rtol = 5e-1
    end
end

@testitem "Basic out rule tests #3" begin
    using ExponentialFamily, ExponentialFamilyProjection, BayesBase

    @testset "f(x) = a*x + b, x ~ Normal (Univariate)" begin
        meta = DeltaMeta(method = CVIProjection(outsamples = 10000), inverse = nothing)
        q_ins = [FactorizedJoint((NormalMeanVariance(0, 2),)), FactorizedJoint((NormalMeanVariance(3, 4),)), FactorizedJoint((NormalMeanVariance(5, 4),))]

        m_out_incomings = [NormalMeanVariance(0, 1), NormalMeanVariance(2, 3), NormalMeanVariance(4, 0.1)]
        constants_b = [-3, -2, -1, 0, 1, 2, 3]
        constants_a = [1, 1.1]
        for q_in in q_ins, a in constants_a, b in constants_b, m_out_incoming in m_out_incomings
            f = (x) -> a * x + b
            q_in_component = first(components(q_in))
            q_out = NormalMeanVariance(a * mean(q_in_component) + b, a^2 * var(q_in_component))

            msg = @call_rule DeltaFn{f}(:out, Marginalisation) (m_out = m_out_incoming, q_out = q_out, q_ins = q_in, meta = meta)

            prj = ProjectedTo(ExponentialFamily.exponential_family_typetag(q_out), size(q_out)...)

            q_out_projected = project_to(prj, (x) -> logpdf(msg, x) + logpdf(m_out_incoming, x))
            @test mean(q_out_projected) ≈ mean(q_out) atol = 5e-1
            @test var(q_out_projected) ≈ var(q_out) atol = 7e-1
        end
    end
end

@testitem "Basic out rule tests #4" begin
    using ExponentialFamily, ExponentialFamilyProjection, BayesBase

    @testset "f(x) = x + constant, x ~ Normal (Multivariate)" begin
        meta = DeltaMeta(method = CVIProjection(), inverse = nothing)

        q_ins_m_out_incomings_fs_constants = [(
            FactorizedJoint((MvNormalMeanCovariance([0.3, 0.7, 10.0], 0.1 * diageye(3)),)),
            MvNormalMeanCovariance(ones(3), 0.9 * diageye(3)),
            x -> x + [0.2, -9.0, 3.0],
            [0.2, -9.0, 3.0]
        )]
        for (q_in, m_out_incoming, f, c) in q_ins_m_out_incomings_fs_constants
            q_in_component = first(components(q_in))
            q_out = MvNormalMeanCovariance(mean(q_in_component) + c, cov(q_in_component))

            msg = @call_rule DeltaFn{f}(:out, Marginalisation) (m_out = m_out_incoming, q_out = q_out, q_ins = q_in, meta = meta)
            prj = ProjectedTo(ExponentialFamily.exponential_family_typetag(q_out), size(q_out)...)
            q_out_projected = project_to(prj, (x) -> logpdf(msg, x), m_out_incoming, initialpoint = q_out)
            @test mean(q_out_projected) ≈ mean(q_out) rtol = 5e-1
            @test var(q_out_projected) ≈ var(q_out) rtol = 5e-1
        end
    end
end

## In these tests we test the Exponential,Pareto, Beta and Gamma distributions with the following non-linearities
## Beta(a,b), f(x) = 1 - x results in Beta(b, a)
## Gamma(a,b), f(x) = cx results in Gamma(a, b*c) in shape scale parameterization
## Exp(λ), f(x) = sqrt(x) results in Rayleigh(1/sqrt(2λ))
## Exp(λ), f(x) = kexp(x) results in Pareto(k,λ) ## EFP errors
## Exp(λ), f(x) = exp(-x) results in Beta(λ, 1) ## EFP errors with NaN
@testitem "Basic out rule tests #5" begin
    using ExponentialFamily, ExponentialFamilyProjection, BayesBase

    @testset "Differen set of non-linearities x ~ NonNormal EF (Univariate)" begin
        meta = DeltaMeta(method = CVIProjection(), inverse = nothing)

        q_ins_m_out_incomings_q_outs_non_linearities = [
            (FactorizedJoint((Beta(5, 2),)), Beta(20, 3), Beta(2, 5), x -> 1 - x),
            (FactorizedJoint((Gamma(3, 4),)), Gamma(10, 7), Gamma(3, 4 * 0.1), x -> 0.1 * x),
            (FactorizedJoint((Exponential(0.5),)), Exponential(3), Rayleigh(1 / (sqrt(2 * 3))), x -> sqrt(x)),
            (FactorizedJoint((Exponential(0.5),)), Exponential(30), Beta(0.5, 1), x -> exp(-x))
        ]
        for (q_in, m_out_incoming, q_out, f) in q_ins_m_out_incomings_q_outs_non_linearities
            msg = @call_rule DeltaFn{f}(:out, Marginalisation) (m_out = m_out_incoming, q_out = q_out, q_ins = q_in, meta = meta)

            prj = ProjectedTo(ExponentialFamily.exponential_family_typetag(q_out), size(q_out)...)

            q_out_projected = project_to(prj, (x) -> logpdf(msg, x) + logpdf(m_out_incoming, x))
            @test mean(q_out_projected) ≈ mean(q_out) atol = 5e-1
            @test var(q_out_projected) ≈ var(q_out) atol = 1e-1
        end
    end
end

@testitem "ProjectionForm target tests" begin
    using ExponentialFamily, ExponentialFamilyProjection, BayesBase, LinearAlgebra

    @testset "Testing abs transformation with ProjectedTo override" begin
        form = ProjectedTo(LogNormal)

        meta = DeltaMeta(method = CVIProjection(out_prjparams = form, outsamples = 1000), inverse = nothing)

        q_in = FactorizedJoint((NormalMeanVariance(0.0, 1.0),))
        m_out = Gamma(2.0, 2.0)
        q_out = Gamma(2.0, 2.0)

        msg = @call_rule DeltaFn{exp}(:out, Marginalisation) (m_out = m_out, q_out = q_out, q_ins = q_in, meta = meta)

        result_dist = msg.numerator
        @test isa(result_dist, LogNormal)

        # For the standard normal, E[exp(x)] = exp(1/2)
        @show result_dist
        @test mean(result_dist) ≈ exp(1 / 2) rtol = 0.05
    end

    @testset "Testing x^2 transformation with MVNormal and ProjectedTo override" begin
        # For 2D case, create form for MVNormal projection
        form = ProjectedTo(
            MvNormalMeanScalePrecision,
            2  # 2-dimensional output
        )

        meta = DeltaMeta(method = CVIProjection(
            out_prjparams = form,
            outsamples = 10000  # More samples for better accuracy
        ), inverse = nothing)

        # Input: Standard 2D normal distribution
        μ = [0.0, 0.0]
        Σ = [1.0 0.5; 0.5 1.0]
        q_in = FactorizedJoint((MvNormalMeanCovariance(μ, Σ),))

        # Output messages (can be arbitrary as they'll be ignored due to ProjectionForm)
        m_out = MvNormalMeanCovariance(ones(2), Matrix{Float64}(I, 2, 2))
        q_out = MvNormalMeanCovariance(ones(2), Matrix{Float64}(I, 2, 2))

        # Function to square each component
        square_components(x) = x .^ 2

        msg = @call_rule DeltaFn{square_components}(:out, Marginalisation) (m_out = m_out, q_out = q_out, q_ins = q_in, meta = meta)

        result_dist = msg.numerator
        @test isa(result_dist, MvNormalMeanScalePrecision)

        # For standard normal X, E[X^2] = 1
        # Each component should have mean ≈ 1
        @test mean(result_dist) ≈ [1.0, 1.0] rtol = 0.1

        # Test with non-zero mean
        μ2 = [7.0, -3.0]
        q_in2 = FactorizedJoint((MvNormalMeanCovariance(μ2, Σ),))

        msg2 = @call_rule DeltaFn{square_components}(:out, Marginalisation) (m_out = m_out, q_out = q_out, q_ins = q_in2, meta = meta)

        result_dist2 = msg2.numerator
        # For normal X with mean μ and variance σ², E[X^2] = μ² + σ²
        expected_means = μ2 .^ 2 .+ diag(Σ)
        @test mean(result_dist2) ≈ expected_means rtol = 0.1
    end
end
