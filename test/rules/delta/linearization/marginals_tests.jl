
@testitem "rules:Delta:linearization:marginals" begin
    using Test, ReactiveMP, BayesBase, Random, ExponentialFamily, Distributions
    import ReactiveMP: @test_marginalrules

    # g: single input, single output
    g(x) = x .^ 2 .- 5

    # h: multiple input, single output
    h(x, y) = x .^ 2 .- y

    @testset "Single univariate input" begin
        @test_marginalrules [check_type_promotion = true] DeltaFn{g}(:ins) [(
            input = (m_out = NormalMeanVariance(2.0, 3.0), m_ins = ManyOf(NormalMeanVariance(2.0, 1.0)), meta = DeltaMeta(; method = Linearization(), inverse = nothing)),
            output = JointNormal(NormalMeanVariance(2.6315789473684212, 0.1578947368421053), ((),))
        )]
    end

    @testset "Single multivariate input" begin
        @test_marginalrules [check_type_promotion = true] DeltaFn{g}(:ins) [(
            input = (
                m_out = MvNormalMeanCovariance([2.0], [3.0]), m_ins = ManyOf(MvNormalMeanCovariance([2.0], [1.0])), meta = DeltaMeta(; method = Linearization(), inverse = nothing)
            ),
            output = JointNormal(MvNormalMeanCovariance([2.6315789473684212], [0.1578947368421053]), ((1,),))
        )]
    end

    @testset "Multiple univairate input" begin
        @test_marginalrules [check_type_promotion = true] DeltaFn{h}(:ins) [(
            input = (
                m_out = NormalMeanVariance(2.0, 3.0),
                m_ins = ManyOf(NormalMeanVariance(2.0, 1.0), NormalMeanVariance(5.0, 1.0)),
                meta  = DeltaMeta(; method = Linearization(), inverse = nothing)
            ),
            output = JointNormal(MvNormalMeanCovariance([2.6, 4.85], [0.20000000000000007 0.19999999999999998; 0.19999999999999998 0.95]), ((), ()))
        )]
    end

    @testset "Multiple multivariate input" begin
        # ForneyLab:test_delta_extended:MDeltaEInGX 2
        @test_marginalrules [check_type_promotion = true] DeltaFn{h}(:ins) [(
            input = (
                m_out = MvNormalMeanCovariance([2.0], [3.0]),
                m_ins = ManyOf(MvNormalMeanCovariance([2.0], [1.0]), MvNormalMeanCovariance([5.0], [1.0])),
                meta  = DeltaMeta(; method = Linearization(), inverse = nothing)
            ),
            output = JointNormal(
                MvNormalMeanCovariance([2.6, 4.85], [0.20000000000000007 0.19999999999999998; 0.19999999999999998 0.95]),
                ((1,), (1,)) # [1, 1] # TODO: dimensions "ds" are not correct for the left marginal
            )
        )]
    end
end # testset
