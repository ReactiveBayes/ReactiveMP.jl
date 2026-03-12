
@testitem "rules:MvNormalMeanScalePrecision:mean" begin
    using ReactiveMP, BayesBase, Random, ExponentialFamily, Distributions

    import ReactiveMP: @test_rules

    @testset "Variational: (q_out::MultivariateNormalDistributionsFamily, q_γ::Gamma)" begin
        @test_rules [check_type_promotion = true] MvNormalMeanScalePrecision(:μ, Marginalisation) [
            (
                input = (q_out = MvNormalMeanCovariance([2.0, 1.0], [3.0 2.0; 2.0 4.0]), q_γ = GammaShapeRate(1.0, 1.0)),
                output = MvNormalMeanScalePrecision([2.0, 1.0], 1.0)
            ),
            (input = (q_out = MvNormalMeanPrecision([2.0, 1.0], [3.0 2.0; 2.0 4.0]), q_γ = Gamma(3.0, 1.0)), output = MvNormalMeanScalePrecision([2.0, 1.0], 3.0))
        ]
    end

    @testset "Structured variational: (m_out::MultivariateNormalDistributionsFamily, q_γ::Gamma)" begin
        @test_rules [check_type_promotion = true] MvNormalMeanScalePrecision(:μ, Marginalisation) [
            (input = (m_out = MvNormalMeanPrecision([2.0, 1.0], [3.0 2.0; 2.0 4.0]), q_γ = Gamma(1.0, 1.0)), output = MvNormalMeanCovariance([2.0, 1.0], [1.5 -0.25; -0.25 1.375])),
            (
                input = (m_out = MvNormalMeanCovariance([0.0, 0.0], [7.0 -1.0; -1.0 9.0]), q_γ = GammaShapeRate(4.0, 2.0)),
                output = MvNormalMeanCovariance([0.0, 0.0], [7.5 -1.0; -1.0 9.5])
            ),
            (
                input = (m_out = MvNormalWeightedMeanPrecision([2.0, 1.0], [3.0 2.0; 2.0 4.0]), q_γ = GammaShapeRate(2.0, 1.0)),
                output = MvNormalMeanCovariance([3 / 4, -1 / 8], [1.0 -0.25; -0.25 0.875])
            )
        ]
    end

    @testset "Structured variational: (m_out::MvNormalMeanScalePrecision, q_γ::Gamma)" begin
        @test_rules [check_type_promotion = true] MvNormalMeanScalePrecision(:μ, Marginalisation) [
            (
                input = (m_out = MvNormalMeanScalePrecision([2.0, 1.0], 3.0), q_γ = Gamma(1.0, 1.0)),
                output = MvNormalMeanScalePrecision([2.0, 1.0], 3.0 * 1.0 / (3.0 + 1.0))
            ),
            (
                input = (m_out = MvNormalMeanScalePrecision([0.0, 0.0], 4.0), q_γ = GammaShapeRate(4.0, 2.0)),
                output = MvNormalMeanScalePrecision([0.0, 0.0], 4.0 * 2.0 / (4.0 + 2.0))
            ),
            (
                input = (m_out = MvNormalMeanScalePrecision([3.0, -1.0], 2.0), q_γ = GammaShapeRate(2.0, 1.0)),
                output = MvNormalMeanScalePrecision([3.0, -1.0], 2.0 * 2.0 / (2.0 + 2.0))
            )
        ]
    end

    @testset "Performance: MvNormalMeanScalePrecision rule allocates less than general rule" begin
        import ReactiveMP: @call_rule

        for n in (10, 100)
            m_out_scale = MvNormalMeanScalePrecision(zeros(n), 3.0)
            m_out_general = MvNormalMeanCovariance(zeros(n), diageye(Float64, n) / 3.0)
            q_γ = GammaShapeRate(2.0, 1.0)

            # Warm up
            @call_rule MvNormalMeanScalePrecision(:μ, Marginalisation) (m_out = m_out_scale, q_γ = q_γ)
            @call_rule MvNormalMeanScalePrecision(:μ, Marginalisation) (m_out = m_out_general, q_γ = q_γ)

            allocs_scale = @allocated @call_rule MvNormalMeanScalePrecision(:μ, Marginalisation) (m_out = m_out_scale, q_γ = q_γ)
            allocs_general = @allocated @call_rule MvNormalMeanScalePrecision(:μ, Marginalisation) (m_out = m_out_general, q_γ = q_γ)

            # The scale rule avoids allocating N×N matrices, so it should allocate at least N times less
            @test allocs_scale * n < allocs_general
        end
    end
end
