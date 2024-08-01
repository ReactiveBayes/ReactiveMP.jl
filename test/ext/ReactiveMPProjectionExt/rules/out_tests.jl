@testitem "Basic out rule tests #1" begin
    using ExponentialFamily, ExponentialFamilyProjection, BayesBase

    ext = Base.get_extension(ReactiveMP, :ReactiveMPProjectionExt)

    @test !isnothing(ext)

    using .ext

    @testset "f(x) = x, x ~ EF" begin
        meta = DeltaMeta(method = CVIProjection(), inverse = nothing)

        q_ins_m_out_incomings = [
            (FactorizedJoint((NormalMeanVariance(2, 2),)), NormalMeanVariance(0, 1)),
            (FactorizedJoint((Gamma(3, 4),)), Gamma(2, 3)),
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

            q_out_projected = project_to(prj, (x) -> logpdf(msg, x) + logpdf(m_out_incoming, x))
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
# - `mean(m_out) ≈ [ mean(m_x); mean(m_y) ]`
@testitem "Basic out rule tests #2" begin
    using ExponentialFamily, ExponentialFamilyProjection, BayesBase, LinearAlgebra
    ext = Base.get_extension(ReactiveMP, :ReactiveMPProjectionExt)
    @test !isnothing(ext)
    using .ext

    meta = DeltaMeta(method = CVIProjection(), inverse = nothing)
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

        q_out_projected = project_to(prj, (x) -> logpdf(msg, x) + logpdf(m_out_incoming, x), initialpoint = q_out)
        @test mean(q_out_projected) ≈ mean(q_out) rtol = 5e-1
        @test var(q_out_projected) ≈ var(q_out) rtol = 1.0
        @test mode(q_out_projected) ≈ mode(q_out) rtol = 5e-1
    end
end

@testitem "Basic out rule tests #3" begin
    using ExponentialFamily, ExponentialFamilyProjection, BayesBase

    ext = Base.get_extension(ReactiveMP, :ReactiveMPProjectionExt)

    @test !isnothing(ext)

    using .ext

    @testset "f(x) = a*x + b, x ~ Normal (Univariate)" begin
        meta = DeltaMeta(method = CVIProjection(), inverse = nothing)
        q_ins = [FactorizedJoint((NormalMeanVariance(0, 2),)), FactorizedJoint((NormalMeanVariance(3, 4),)), FactorizedJoint((NormalMeanVariance(5, 4),))]

        m_out_incomings = [NormalMeanVariance(0, 1), NormalMeanVariance(2, 3), NormalMeanVariance(4, 0.1)]
        constants_b = [-2, -1, 0, 1, 2]
        constants_a = [1, 1.1, 2]
        for q_in in q_ins, a in constants_a, b in constants_b, m_out_incoming in m_out_incomings
            f = (x) -> a * x + b
            q_in_component = first(components(q_in))
            q_out = NormalMeanVariance(a * mean(q_in_component) + b, a^2 * var(q_in_component))

            msg = @call_rule DeltaFn{f}(:out, Marginalisation) (m_out = m_out_incoming, q_out = q_out, q_ins = q_in, meta = meta)

            prj = ProjectedTo(ExponentialFamily.exponential_family_typetag(q_out), size(q_out)...)

            q_out_projected = project_to(prj, (x) -> logpdf(msg, x) + logpdf(m_out_incoming, x); initialpoint = q_out)
            @test mean(q_out_projected) ≈ mean(q_out) atol = 5e-1
            @test var(q_out_projected) ≈ var(q_out) atol = 2
        end
    end
end

@testitem "Basic out rule tests #4" begin
    using ExponentialFamily, ExponentialFamilyProjection, BayesBase

    ext = Base.get_extension(ReactiveMP, :ReactiveMPProjectionExt)

    @test !isnothing(ext)

    using .ext

    @testset "f(x) = x + constant, x ~ Normal (Multivariate)" begin
        meta = DeltaMeta(method = CVIProjection(), inverse = nothing)

        q_ins_m_out_incomings_fs_constants = [
            (FactorizedJoint((MvNormalMeanCovariance(zeros(3), 2 * diageye(3)),)), MvNormalMeanCovariance(zeros(3), 0.4 * diageye(3)), x -> x + [1.0, 2.0, 0.4], [1.0, 2.0, 0.4]),
            (
                FactorizedJoint((MvNormalMeanCovariance([0.3, 0.7, 10.0], 0.1 * diageye(3)),)),
                MvNormalMeanCovariance(ones(3), 0.9 * diageye(3)),
                x -> x + [0.2, -9.0, 3.0],
                [0.2, -9.0, 3.0]
            )
        ]
        for (q_in, m_out_incoming, f, c) in q_ins_m_out_incomings_fs_constants
            q_in_component = first(components(q_in))
            q_out = MvNormalMeanCovariance(mean(q_in_component) + c, cov(q_in_component))

            msg = @call_rule DeltaFn{f}(:out, Marginalisation) (m_out = m_out_incoming, q_out = q_out, q_ins = q_in, meta = meta)
            prj = ProjectedTo(ExponentialFamily.exponential_family_typetag(q_out), size(q_out)...)

            q_out_projected = project_to(prj, (x) -> logpdf(msg, x), m_out_incoming, initialpoint = q_out)

            @test mean(q_out_projected) ≈ mean(q_out) rtol = 5e-1
        end
    end
end

@testitem "Basic out rule tests #5" begin
    using ExponentialFamily, ExponentialFamilyProjection, BayesBase, LinearAlgebra

    ext = Base.get_extension(ReactiveMP, :ReactiveMPProjectionExt)

    @test !isnothing(ext)

    using .ext

    @testset "f(x) = Ax , x ~ Normal (Multivariate)" begin
        meta = DeltaMeta(method = CVIProjection(), inverse = nothing)

        q_ins_m_out_incomings_fs_constants = [
            (
                FactorizedJoint((MvNormalMeanCovariance(ones(3), 2 * diageye(3)),)),
                MvNormalMeanCovariance(zeros(3), 0.4 * diageye(3)),
                x -> [1.0 2.0 0.4; 0.1 0.2 0.3; 0.1 -0.3 0.5] * x,
                [1.0 2.0 0.4; 0.1 0.2 0.3; 0.1 -0.3 0.5]
            ),
            (
                FactorizedJoint((MvNormalMeanCovariance([0.3, 0.7, 10.0], 0.1 * diageye(3)),)),
                MvNormalMeanCovariance(ones(3), 0.9 * diageye(3)),
                x -> [1.6 0.2 0.4; -0.1 0.2 -0.3; 0.1 -0.3 0.5] * x,
                [1.6 0.2 0.4; -0.1 0.2 -0.3; 0.1 -0.3 0.5]
            )
        ]
        for (q_in, m_out_incoming, f, c) in q_ins_m_out_incomings_fs_constants
            q_in_component = first(components(q_in))
            q_out = MvNormalMeanCovariance(c * mean(q_in_component), c * cov(q_in_component) * c')

            msg = @call_rule DeltaFn{f}(:out, Marginalisation) (m_out = m_out_incoming, q_out = q_out, q_ins = q_in, meta = meta)
            prj = ProjectedTo(ExponentialFamily.exponential_family_typetag(q_out), size(q_out)...)

            q_out_projected = project_to(prj, (x) -> logpdf(msg, x), m_out_incoming, initialpoint = q_out)

            @test mean(q_out_projected) ≈ mean(q_out) rtol = 5e-1
        end
    end
end

## In these tests we test the Exponential,Pareto, Beta and Gamma distributions with the following non-linearities
## Beta(a,b), f(x) = 1 - x results in Beta(b, a)
## Gamma(a,b), f(x) = cx results in Gamma(a, b*c) in shape scale parameterization
## Exp(λ), f(x) = sqrt(x) results in Rayleigh(1/sqrt(2λ))
## Exp(λ), f(x) = kexp(x) results in Pareto(k,λ) ## EFP errors
## Exp(λ), f(x) = exp(-x) results in Beta(λ, 1) 
@testitem "Basic out rule tests #6" begin
    using ExponentialFamily, ExponentialFamilyProjection, BayesBase

    ext = Base.get_extension(ReactiveMP, :ReactiveMPProjectionExt)

    @test !isnothing(ext)

    using .ext

    @testset "Differen set of non-linearities x ~ NonNormal EF (Univariate)" begin
        projection_optional = CVIProjectionOptional(out_samples_no = 5000)
        meta = DeltaMeta(method = CVIProjection(projection_optional = projection_optional), inverse = nothing)

        q_ins_m_out_incomings_q_outs_non_linearities = [
            (FactorizedJoint((Beta(5, 2),)), Beta(20, 3), Beta(2, 5), x -> 1 - x),
            (FactorizedJoint((Gamma(3, 4),)), Gamma(10, 7), Gamma(3, 4 * 0.1), x -> 0.1 * x),
            (FactorizedJoint((Exponential(0.5),)), Exponential(3), Rayleigh(1 / (sqrt(2 * 3))), x -> sqrt(x)),
            # (FactorizedJoint((Exponential(0.5),)), Exponential(3), Geometric(1 - exp(-0.5)), x->ceil(x)) ,
            # (FactorizedJoint((Exponential(0.5),)), Exponential(3), Pareto(3, 0.5), x->3*exp(x) ),
            (FactorizedJoint((Exponential(0.5),)), Exponential(3), Beta(7, 1), x -> exp(-x))
        ]
        for (q_in, m_out_incoming, q_out, f) in q_ins_m_out_incomings_q_outs_non_linearities
            msg = @call_rule DeltaFn{f}(:out, Marginalisation) (m_out = m_out_incoming, q_out = q_out, q_ins = q_in, meta = meta)
            prj_parameters = ProjectionParameters(niterations = 10)
            prj = ProjectedTo(ExponentialFamily.exponential_family_typetag(q_out), size(q_out)...; parameters = prj_parameters)

            q_out_projected = project_to(prj, (x) -> logpdf(msg, x) + logpdf(m_out_incoming, x))
            @test mean(q_out_projected) ≈ mean(q_out) atol = 5e-1
            @test var(q_out_projected) ≈ var(q_out) atol = 5e-1
        end
    end
end

@testitem "Basic out rule tests #7: Generic sum-product projection" begin
    using ExponentialFamily, ExponentialFamilyProjection, BayesBase, LinearAlgebra
    import ExponentialFamily: WishartFast
    using ForwardDiff
    ext = Base.get_extension(ReactiveMP, :ReactiveMPProjectionExt)

    @test !isnothing(ext)

    using .ext

    @testset "Identity non-linearities univariate in univariate out " begin
        projection_types = (out = Gamma, in = (Gamma,))
        projection_dimensions = (out = (), in = ((),))
        projection_essentials = CVIProjectionEssentials(projection_types = projection_types, projection_dims = projection_dimensions)
        projection_optionals = CVIProjectionOptional(out_samples_no = 2000)
        meta = DeltaMeta(method = CVIProjection(projection_essentials = projection_essentials, projection_optional = projection_optionals), inverse = nothing)

        for a in (3.0:6.0), b in (1.2:2.2)
            msg_in  = Gamma(a, b)
            msg_out = @call_rule DeltaFn{identity}(:out, Marginalisation) (m_ins = ManyOf(msg_in), meta = meta)

            @test mean(msg_out) ≈ mean(msg_in) atol = 5e-1
        end
    end

    @testset "Differen set of non-linearities Multivariate in Multivariate out " begin
        projection_types = (out = MvNormalMeanCovariance, in = (MvNormalMeanCovariance,))
        projection_dimensions = (out = (2,), in = ((2,),))
        projection_essentials = CVIProjectionEssentials(projection_types = projection_types, projection_dims = projection_dimensions)
        projection_optionals = CVIProjectionOptional(out_samples_no = 2000)
        meta = DeltaMeta(method = CVIProjection(projection_essentials = projection_essentials, projection_optional = projection_optionals), inverse = nothing)

        f = (x, k) -> x .^ k
        for m in (zeros(2), ones(2), randn(2)), v in ([1, 2], [0.1, 0.8], [4.0, 0.2]), k in (1, 2, 3)
            msg_in = MvNormalMeanCovariance(m, Diagonal(v))
            _f = (x) -> f(x, k)
            msg_out = @call_rule DeltaFn{_f}(:out, Marginalisation) (m_ins = ManyOf(msg_in), meta = meta)
            ### We know th exact solution of this upto arbitrary precision but for now very high tolerance
            @test mean(msg_out) ≈ m .^ k rtol = 5
        end
    end

    @testset "Differen set of non-linearities multi-univariate in in multivariate out " begin
        projection_types = (out = MvNormalMeanCovariance, in = (MvNormalMeanCovariance, Gamma))
        projection_dimensions = (out = (2,), in = ((2,), ()))
        projection_essentials = CVIProjectionEssentials(projection_types = projection_types, projection_dims = projection_dimensions)
        meta = DeltaMeta(method = CVIProjection(projection_essentials = projection_essentials), inverse = nothing)

        f = (x, y) -> y * x
        msg_out = @call_rule DeltaFn{f}(:out, Marginalisation) (m_ins = ManyOf(MvNormalMeanCovariance(zeros(2), diageye(2)), Gamma(3, 3)), meta = meta)

        @test mean(msg_out) ≈ zeros(2) rtol = 3.0
    end

    @testset "Differen set of non-linearities multi-multivariate in in multivariate out " begin
        projection_types = (out = MvNormalMeanCovariance, in = (MvNormalMeanCovariance, MvNormalMeanCovariance))
        projection_dimensions = (out = (2,), in = ((2,), (2,)))
        projection_essentials = CVIProjectionEssentials(projection_types = projection_types, projection_dims = projection_dimensions)
        meta = DeltaMeta(method = CVIProjection(projection_essentials = projection_essentials), inverse = nothing)

        f = (x, y) -> x + y
        msg_out = @call_rule DeltaFn{f}(:out, Marginalisation) (
            m_ins = ManyOf(MvNormalMeanCovariance(zeros(2), diageye(2)), MvNormalMeanCovariance(ones(2), diageye(2))), meta = meta
        )

        @test mean(msg_out) ≈ ones(2) rtol = 5e-1
    end

    @testset "Differen set of non-linearities univariate in univariate out " begin
        projection_types = (out = Gamma, in = (Gamma, Beta))
        projection_dimensions = (out = (), in = ((), ()))
        projection_essentials = CVIProjectionEssentials(projection_types = projection_types, projection_dims = projection_dimensions)
        meta = DeltaMeta(method = CVIProjection(projection_essentials = projection_essentials), inverse = nothing)

        f = (x, y) -> x + y
        msg = @call_rule DeltaFn{f}(:out, Marginalisation) (m_ins = ManyOf(Gamma(3, 3), Beta(3, 4)), meta = meta)
        @test mean(msg) ≈ mean(Gamma(3, 3)) + mean(Beta(3, 4)) atol = 1
    end

    @testset "Differen set of non-linearities univariate in multivariate out " begin
        projection_types = (out = MvNormalMeanCovariance, in = (Gamma, Beta))
        projection_dimensions = (out = (2,), in = ((), ()))
        projection_essentials = CVIProjectionEssentials(projection_types = projection_types, projection_dims = projection_dimensions)
        meta = DeltaMeta(method = CVIProjection(projection_essentials = projection_essentials), inverse = nothing)

        f   = (x, y) -> [x, y]
        msg = @call_rule DeltaFn{f}(:out, Marginalisation) (m_ins = ManyOf(Gamma(3, 3), Beta(3, 4)), meta = meta)

        @test mean(msg) ≈ f(mean(Gamma(3, 3)), mean(Beta(3, 4))) rtol = 5e-1
    end

    # @testset "Differen set of non-linearities univariate ins matrix variate out" begin
    #     projection_types = (out = WishartFast, in = (Gamma, NormalMeanVariance))
    #     projection_dimensions = (out = (2,2), in =((),()))
    #     projection_essentials = CVIProjectionEssentials(projection_types = projection_types, projection_dims = projection_dimensions)
    #     meta = DeltaMeta(method = CVIProjection(projection_essentials = projection_essentials), inverse = nothing)

    #     f    = (x, y) -> [x -y; -y x]
    #     msg  = @call_rule DeltaFn{f}(:out, Marginalisation) (m_ins = ManyOf(Gamma(3, 3), NormalMeanVariance(0, 1e-7)), meta = meta)

    #     @test mean(msg) ≈ [mean(Gamma(3,3)) 0; 0 mean(Gamma(3,3)) ] rtol = 5e-1
    # end

    # @testset "Different set of non-linearities matrix-univariate in matrixvariate out " begin
    #     projection_types = (out = MvNormalMeanCovariance, in = (WishartFast, Beta))
    #     projection_dimensions = (out = (2,2), in =((2,2),()))
    #     projection_essentials = CVIProjectionEssentials(projection_types = projection_types, projection_dims = projection_dimensions)
    #     meta = DeltaMeta(method = CVIProjection(projection_essentials = projection_essentials), inverse = nothing)

    #     f    = (x, y) -> x*y

    #     msg  = @call_rule DeltaFn{f}(:out, Marginalisation) (m_ins = ManyOf(WishartFast(5, Diagonal(ones(2))), Beta(3, 4)), meta = meta)

    #     # @show msg([vec(V); 0.2])
    # end
end

@testitem "Basic out rule tests #8: Generic sum-product projection" begin
    using ExponentialFamily, ExponentialFamilyProjection, BayesBase

    ext = Base.get_extension(ReactiveMP, :ReactiveMPProjectionExt)

    @test !isnothing(ext)

    using .ext
    @testset "Beta with f(x) = 1 - x " begin
        projection_types = (out = Beta, in = (Beta,))
        projection_dimensions = (out = (), in = ((),))
        projection_essentials = CVIProjectionEssentials(projection_types = projection_types, projection_dims = projection_dimensions)
        meta = DeltaMeta(method = CVIProjection(projection_essentials = projection_essentials), inverse = nothing)

        f = (x) -> 1 - x
        for a in 3:6, b in 4:9
            msg_in = Beta(a, b)
            msg = @call_rule DeltaFn{f}(:out, Marginalisation) (m_ins = ManyOf(msg_in), meta = meta)

            @test mean(msg) ≈ mean(Beta(b, a)) atol = 5e-1
            @test var(msg) ≈ var(Beta(b, a)) atol = 5e-1
            @test collect(params(msg)) ≈ [b, a] rtol = 5e-1
        end
    end

    @testset "Gamma with f(x) = cx " begin
        projection_types = (out = Gamma, in = (Gamma,))
        projection_dimensions = (out = (), in = ((),))
        projection_essentials = CVIProjectionEssentials(projection_types = projection_types, projection_dims = projection_dimensions)
        meta = DeltaMeta(method = CVIProjection(projection_essentials = projection_essentials), inverse = nothing)
        for c in (0.1:0.1:2), a in (2, 3), b in (0.1, 0.5) ## if b is high then the mean and move gets large and tolerance needs to be adaptively adjusted
            f   = (x) -> c * x
            msg = @call_rule DeltaFn{f}(:out, Marginalisation) (m_ins = ManyOf(Gamma(a, b)), meta = meta)

            @test mean(msg) ≈ mean(Gamma(a, b * c)) atol = 5e-1
            @test var(msg) ≈ var(Gamma(a, b * c)) atol = 5e-1
        end
    end
end

@testitem "Basic out rule tests #9" begin
    using ExponentialFamily, ExponentialFamilyProjection, BayesBase

    ext = Base.get_extension(ReactiveMP, :ReactiveMPProjectionExt)

    @test !isnothing(ext)

    using .ext

    @testset "Different set of non-linearities x ~ NonNormal EF (Univariate)" begin
        projection_optional = CVIProjectionOptional(out_samples_no = 5000)
        projection_types = (out = NormalMeanVariance, in = (Gamma, Beta, NormalMeanVariance))
        projection_dimensions = (out = (), in = ((), (), ()))
        projection_essentials = CVIProjectionEssentials(projection_types = projection_types, projection_dims = projection_dimensions)
        meta = DeltaMeta(method = CVIProjection(projection_essentials = projection_essentials, projection_optional = projection_optional), inverse = nothing)
        f = (x, y, z) -> x * y + z
        m_in1 = Gamma(3, 4)
        m_in2 = Beta(5, 2)
        m_in3 = NormalMeanVariance(3, 0.2)
        m_out_incoming = NormalMeanVariance(0.0, 1.0)

        msg_sum_product_projection = @call_rule DeltaFn{f}(:out, Marginalisation) (m_ins = ManyOf(m_in1, m_in2, m_in3), meta = meta)
        q_in = FactorizedJoint((m_in1, m_in2, m_in3))

        msg = @call_rule DeltaFn{f}(:out, Marginalisation) (m_out = m_out_incoming, q_out = msg_sum_product_projection, q_ins = q_in, meta = meta)
        prj_parameters = ProjectionParameters(niterations = 1000)
        prj = ProjectedTo(ExponentialFamily.exponential_family_typetag(msg_sum_product_projection), size(msg_sum_product_projection)...; parameters = prj_parameters)

        q_out_projected = project_to(prj, (x) -> logpdf(msg, x) + logpdf(m_out_incoming, x))
        @test mean(q_out_projected) ≈ mean(msg_sum_product_projection) atol = 5e-1
    end
end
