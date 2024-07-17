@testitem "Basic checks for marginal rule" begin
    using ExponentialFamily, ExponentialFamilyProjection, BayesBase
    import ReactiveMP: @test_rules, @test_marginalrules

    ext = Base.get_extension(ReactiveMP, :ReactiveMPProjectionExt)

    @test !isnothing(ext)

    using .ext

    # So here we use an `identity` function, in which case the delta node is a no-op
    @testset "f(x) -> x, x~EF, out~EF" begin
        meta = DeltaMeta(method = CVIProjection(), inverse = nothing)
        # Since we use `identity` as a function we expect that the result of marginal computation is a product of `m_out` and `m_in`
        inputs = [
            (NormalMeanVariance(0, 1), NormalMeanVariance(0, 1)),
            (Gamma(2, 2), Gamma(2, 2)),
            (Beta(1, 1), Beta(1, 1)),
            (MvNormalMeanCovariance([0.5, 0.5]), MvNormalMeanCovariance([0.5, 0.5])),
            (MvNormalMeanCovariance([0.5, 0.5, -1.0]), MvNormalMeanCovariance([0.5, 2.5, -3.0]))
        ]
        for input in inputs
            m_in = input[1]
            m_out = input[1]
            q_factorised = @call_marginalrule DeltaFn{identity}(:ins) (m_out = m_out, m_ins = ManyOf(m_in), meta = meta)
            @test length(q_factorised) === 1
            q_in_1 = component(q_factorised, 1)
            @test q_in_1 ≈ prod(GenericProd(), m_in, m_out) atol = 1e-1
        end
    end

    @testset "f(x, y) -> [x, y], x~Normal, y~Normal, out~MvNormal (marginalization)" begin
        f(x, y) = [x, y]
        meta = DeltaMeta(method = CVIProjection(), inverse = nothing)
        @test_marginalrules [check_type_promotion = false, atol = 1e-2] DeltaFn{f}(:ins) [(
            input = (m_out = MvGaussianMeanCovariance(ones(2), [2 0; 0 2]), m_ins = ManyOf(NormalMeanVariance(0, 1), NormalMeanVariance(1, 2)), meta = meta),
            output = FactorizedJoint((NormalMeanVariance(1 / 3, 2 / 3), NormalMeanVariance(1.0, 1.0)))
        )]
    end
end