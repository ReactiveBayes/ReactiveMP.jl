@testitem "Basic checks for marginal rule" begin
    using ExponentialFamily, ExponentialFamilyProjection, BayesBase, SpecialFunctions
    import ReactiveMP: @test_rules, @test_marginalrules

    ext = Base.get_extension(ReactiveMP, :ReactiveMPProjectionExt)

    @test !isnothing(ext)

    using .ext

    # @testset "f(x) -> x, x~EF, out~EF" begin
    #     meta = DeltaMeta(method = CVIProjection(), inverse = nothing)
    #     # Since we use `identity` as a function we expect that the result of marginal computation is a product of `m_out` and `m_in`
    #     inputs = [
    #         NormalMeanVariance(0, 1),
    #         Gamma(2, 2),
    #         Beta(1, 1), 
    #         MvNormalMeanCovariance([0.5, 0.5]),
    #         MvNormalMeanCovariance([0.5, 0.5, -1.0]), 
    #     ]
    #     for input in inputs
    #         m_in = input
    #         m_out = input
    #         q_factorised = @call_marginalrule DeltaFn{identity}(:ins) (m_out = m_out, m_ins = ManyOf(m_in), meta = meta)
    #         @test length(q_factorised) === 1
    #         q_in_1 = component(q_factorised, 1)
    #         @test q_in_1 ≈ prod(GenericProd(), m_in, m_out) atol = 1e-1
    #     end
    # end

    # @testset "f(x) -> x, x~EF, out~EF with conditioners" begin
    #     meta = DeltaMeta(method = CVIProjection(), inverse = nothing)
    #     # Since we use `identity` as a function we expect that the result of marginal computation is a product of `m_out` and `m_in`
    #     inputs = [
    #         Binomial(10, 0.4),
    #         Poisson(4.0)
    #     ]
    #     for input in inputs
    #         m_in = input
    #         m_out = input
    #         q_factorised = @call_marginalrule DeltaFn{identity}(:ins) (m_out = m_out, m_ins = ManyOf(m_in), meta = meta)
    #         @test length(q_factorised) === 1
    #         q_ef_component = convert(ExponentialFamilyDistribution, component(q_factorised, 1))
    #         q_prod = prod(PreserveTypeProd(ExponentialFamilyDistribution), m_in, m_out)
    #         η_prod = getnaturalparameters(q_prod)
    #         @test_broken getnaturalparameters(q_ef_component) ≈ η_prod atol = 5e-1    
    #     end
    # end

    # @testset "f(x, y) -> [x, y], x~Normal, y~Normal, out~MvNormal (marginalization)" begin
    #     f(x, y) = [x, y]
    #     projection_optional = CVIProjectionOptional(marginal_samples_no = (5000,5))
    #     meta = DeltaMeta(method = CVIProjection(projection_optional = projection_optional), inverse = nothing)
    #     @test_marginalrules [check_type_promotion = false, atol = 1e-1] DeltaFn{f}(:ins) [(
    #         input = (m_out = MvGaussianMeanCovariance(ones(2), [2 0; 0 2]), m_ins = ManyOf(NormalMeanVariance(0, 1), NormalMeanVariance(1, 2)), meta = meta),
    #         output = FactorizedJoint((NormalMeanVariance(1 / 3, 2 / 3), NormalMeanVariance(1.0, 1.0)))
    #     )]
    # end

    # @testset "f(x, y) -> [x, y], x~function, y~function, out~MvMvNormal (marginalization)" begin
    #     f(x, y) = [x, y]
    #     projection_types = (out = MvNormalMeanCovariance, in = (NormalMeanVariance, NormalMeanVariance))
    #     projection_dimensions = (out = (2,), in = ((), ()))
    #     projection_optional = CVIProjectionOptional(marginal_samples_no = (5000,5)) #because the tolerance is atol we use high number of samples
    #     projection_essentials = CVIProjectionEssentials(projection_types = projection_types, projection_dims = projection_dimensions)
    #     meta = DeltaMeta(method = CVIProjection(projection_essentials = projection_essentials, projection_optional = projection_optional), inverse = nothing)
    #     @test_marginalrules [check_type_promotion = false, atol = 1e-1] DeltaFn{f}(:ins) [(
    #         input = (
    #             m_out = MvGaussianMeanCovariance(ones(2), [2 0; 0 2]),
    #             m_ins = ManyOf(x -> logpdf(NormalMeanVariance(0, 1), x), x -> logpdf(NormalMeanVariance(1, 2), x)),
    #             meta = meta
    #         ),
    #         output = FactorizedJoint((NormalMeanVariance(1 / 3, 2 / 3), NormalMeanVariance(1.0, 1.0)))
    #     )]
    # end

    # @testset "f(x, y) -> [x, y], x~MvNormal, y~MvNormal, out~MvMvNormal (marginalization)" begin
    #     f(x, y) = [x; y]
    #     projection_types = (out = MvNormalMeanCovariance, in = (NormalMeanVariance, NormalMeanVariance))
    #     projection_dimensions = (out = (5,), in = ((2,), (3,)))
    #     projection_optional = CVIProjectionOptional(marginal_samples_no = (5000, 5)) 
    #     projection_essentials = CVIProjectionEssentials(projection_types = projection_types, projection_dims = projection_dimensions)
    #     meta = DeltaMeta(method = CVIProjection(projection_essentials = projection_essentials, projection_optional = projection_optional), inverse = nothing)
    #     @test_marginalrules [check_type_promotion = false, atol = 1e-1] DeltaFn{f}(:ins) [(
    #         input = (
    #             m_out = MvGaussianMeanCovariance(ones(5), 2 * diageye(5)),
    #             m_ins = ManyOf(MvNormalMeanCovariance(zeros(2), diageye(2)), MvNormalMeanCovariance(ones(3), 2 * diageye(3))),
    #             meta = meta
    #         ),
    #         output = FactorizedJoint((MvNormalMeanCovariance(1 / 3 * ones(2), 2 / 3 * diageye(2)), MvNormalMeanCovariance(ones(3), diageye(3))))
    #     )]
    # end

    # @testset "f(x, y) -> y + x, x~Poisson, y~Poisson, out~Poisson (marginalization)" begin
    #     f(x, y) = x + y
    #     projection_types = (out = Poisson, in = (Poisson, Poisson))
    #     projection_dimensions = (out = (), in = ((), ()))
    #     projection_optional = CVIProjectionOptional(marginal_samples_no = (2000, 5)) ## because the tolerance is atol we use high number of samples
    #     projection_essentials = CVIProjectionEssentials(projection_types = projection_types, projection_dims = projection_dimensions)
    #     meta = DeltaMeta(method = CVIProjection(projection_essentials = projection_essentials, projection_optional = projection_optional), inverse = nothing)
    #     for λout in (2, 5) , λin1 in (3, 4), λin2 in (6, 7)
    #         m_out = Poisson(λout)
    #         m_in1 = Poisson(λin1)
    #         m_in2 = Poisson(λin2)
    #         marginal = @call_marginalrule DeltaFn{f}(:ins) (m_out = m_out, m_ins = ManyOf(m_in1, m_in2), meta = meta)
    #         marginal_components = components(marginal)
    #         component1 = first(marginal_components)
    #         component2 = last(marginal_components)
    #         ## This is a joint pdf. Our method approximates this joint with independent marginals as factorized joint.
    #         exact_marginal_logpdf = (x) -> x[1]*(log(λout) + log(λin1)) + x[2]*(log(λout) + log(λin2)) - (loggamma(x[1] + 1) + loggamma(x[2] +1)  + loggamma(x[1]+x[2]+1) )
    #         exact_marginal_pdf_unnormalized = x -> exp(exact_marginal_logpdf(x))

    #         grid = [[x,y] for x in 0:100, y in 0:100]
    #         normalization = sum(map(exact_marginal_pdf_unnormalized, grid))
    #         exact_marginal_pdf_normalized = x -> exact_marginal_pdf_unnormalized(x)/normalization

    #         exact_mean = mapreduce(x -> x*exact_marginal_pdf_normalized(x), +, grid)

    #         @test [mean(component1), mean(component2)] ≈ exact_mean rtol = 5e-1
            
    #     end
    # end

    # @testset "f(x, y) -> [1 - x, 1 - y], x~Beta, y~Beta, out~FactorizedJoint((Beta, Beta)) (marginalization)" begin
    #     f(x, y) = [1 - x, 1 - y]
    #     projection_types = (out = FactorizedJoint, in = (Beta, Beta))
    #     projection_dimensions = (out = (2,), in = ((), ()))
    #     projection_optional = CVIProjectionOptional(marginal_samples_no = (1000, 5))
    #     projection_essentials = CVIProjectionEssentials(projection_types = projection_types, projection_dims = projection_dimensions)
    #     meta = DeltaMeta(method = CVIProjection(projection_essentials = projection_essentials, projection_optional = projection_optional), inverse = nothing)
    #     for a1 in (2, 3), b1 in (2, 3), a2 in (3, 4), b2 in (4, 5), ain1 in (3, 5), ain2 in (4, 5), bin1 in (2, 4), bin2 in (4, 8)
    #         m_out1 = Beta(a1, b1)
    #         m_out2 = Beta(a2, b2)
    #         m_in1 = Beta(ain1, bin1)
    #         m_in2 = Beta(ain1, bin2)
    #         expected_out1 = prod(GenericProd(), Beta(b1, a1), Beta(ain1, bin1))
    #         expected_out2 = prod(GenericProd(), Beta(b2, a2), Beta(ain2, bin2))

    #         marginal_out = @call_marginalrule DeltaFn{f}(:ins) (m_out = FactorizedJoint((m_out1, m_out2)), m_ins = ManyOf(m_in1, m_in2), meta = meta)
    #         marginal_components = components(marginal_out)

    #         marginal_component1 = first(marginal_components)
    #         marginal_component2 = last(marginal_components)

    #         @test collect(params(marginal_component1)) ≈ collect(params(expected_out1)) rtol = 5e-1
    #         @test collect(params(marginal_component2)) ≈ collect(params(expected_out2)) rtol = 5e-1
    #         @test mean(marginal_component1) ≈ mean(expected_out1) atol = 5e-1
    #         @test mean(marginal_component2) ≈ mean(expected_out2) atol = 5e-1
    #     end
    # end
end