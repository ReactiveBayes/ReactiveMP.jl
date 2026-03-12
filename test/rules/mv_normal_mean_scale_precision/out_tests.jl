
@testitem "rules:MvNormalMeanPrecision:out" begin
    using ReactiveMP, BayesBase, Random, ExponentialFamily, Distributions

    import ReactiveMP: @test_rules

    @testset "Variational: (q_out::MultivariateNormalDistributionsFamily, q_γ::Gamma)" begin
        @test_rules [check_type_promotion = true] MvNormalMeanScalePrecision(:out, Marginalisation) [
            (input = (q_μ = MvNormalMeanCovariance([2.0, 1.0], [3.0 2.0; 2.0 4.0]), q_γ = Gamma(1.0, 1.0)), output = MvNormalMeanScalePrecision([2.0, 1.0], 1.0)),
            (input = (q_μ = MvNormalMeanPrecision([2.0, 1.0], [3.0 2.0; 2.0 4.0]), q_γ = Gamma(3.0, 2.0)), output = MvNormalMeanScalePrecision([2.0, 1.0], 6.0)),
            (input = (q_μ = MvNormalWeightedMeanPrecision([2.0, 1.0], [3.0 2.0; 2.0 4.0]), q_γ = Gamma(4.0, 2.0)), output = MvNormalMeanScalePrecision([3 / 4, -1 / 8], 8.0))
        ]
    end

    @testset "Structured variational: (m_μ::MultivariateNormalDistributionsFamily, q_γ::Gamma)" begin
        @test_rules [check_type_promotion = true] MvNormalMeanScalePrecision(:out, Marginalisation) [
            (input = (m_μ = MvNormalMeanCovariance([2.0, 1.0], [3.0 2.0; 2.0 4.0]), q_γ = Gamma(2.0, 1.0)), output = MvNormalMeanCovariance([2.0, 1.0], [3.5 2.0; 2.0 4.5])),
            (input = (m_μ = MvNormalMeanCovariance([0.0, 1.0], [2.0 -1.0; -1.0 4.0]), q_γ = Gamma(3.0, 1.0)), output = MvNormalMeanCovariance([0.0, 1.0], [7/3 -1.0; -1.0 13/3])),
            (
                input = (m_μ = MvNormalWeightedMeanPrecision([3.0, -1.0], [1.0 0.0; 0.0 1.0]), q_γ = Gamma(4.0, 2.0)),
                output = MvNormalMeanCovariance([3.0, -1.0], [1.125 0.0; 0.0 1.125])
            )
        ]
    end

    @testset "Structured variational: (m_μ::MvNormalMeanScalePrecision, q_γ::Gamma)" begin
        @test_rules [check_type_promotion = true] MvNormalMeanScalePrecision(:out, Marginalisation) [
            (input = (m_μ = MvNormalMeanScalePrecision([2.0, 1.0], 3.0), q_γ = Gamma(1.0, 1.0)), output = MvNormalMeanScalePrecision([2.0, 1.0], 3.0 * 1.0 / (3.0 + 1.0))),
            (input = (m_μ = MvNormalMeanScalePrecision([0.0, 0.0], 4.0), q_γ = GammaShapeRate(4.0, 2.0)), output = MvNormalMeanScalePrecision([0.0, 0.0], 4.0 * 2.0 / (4.0 + 2.0))),
            (
                input = (m_μ = MvNormalMeanScalePrecision([3.0, -1.0], 2.0), q_γ = GammaShapeRate(2.0, 1.0)),
                output = MvNormalMeanScalePrecision([3.0, -1.0], 2.0 * 2.0 / (2.0 + 2.0))
            )
        ]
    end

    @testset "Performance: MvNormalMeanScalePrecision rule allocates less than general rule" begin
        import ReactiveMP: @call_rule

        for n in (10, 100)
            m_μ_scale = MvNormalMeanScalePrecision(zeros(n), 3.0)
            m_μ_general = MvNormalMeanCovariance(zeros(n), diageye(Float64, n) / 3.0)
            q_γ = GammaShapeRate(2.0, 1.0)

            # Warm up
            @call_rule MvNormalMeanScalePrecision(:out, Marginalisation) (m_μ = m_μ_scale, q_γ = q_γ)
            @call_rule MvNormalMeanScalePrecision(:out, Marginalisation) (m_μ = m_μ_general, q_γ = q_γ)

            allocs_scale = @allocated @call_rule MvNormalMeanScalePrecision(:out, Marginalisation) (m_μ = m_μ_scale, q_γ = q_γ)
            allocs_general = @allocated @call_rule MvNormalMeanScalePrecision(:out, Marginalisation) (m_μ = m_μ_general, q_γ = q_γ)

            # The scale rule avoids allocating N×N matrices, so it should allocate at least N times less
            @test allocs_scale * n < allocs_general
        end
    end
end
