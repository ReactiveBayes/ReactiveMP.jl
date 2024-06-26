
@testitem "rules:typeof(dot):out" begin
    using ReactiveMP, BayesBase, Random, ExponentialFamily, Distributions

    import ReactiveMP: @test_rules
    import LinearAlgebra: dot
    import MatrixCorrectionTools: NoCorrection, ReplaceZeroDiagonalEntries

    @testset "Belief Propagation: (m_in1::PointMass, m_in2::NormalDistributionsFamily)" begin
        @test_rules [check_type_promotion = true] typeof(dot)(:out, Marginalisation) [
            (input = (m_in1 = PointMass(1.0), m_in2 = NormalMeanVariance(2.0, 2.0), meta = NoCorrection()), output = NormalMeanVariance(2.0, 2.0)),
            (input = (m_in1 = PointMass(-1.0), m_in2 = NormalMeanPrecision(3.0, 1.0), meta = NoCorrection()), output = NormalMeanVariance(-3.0, 1.0)),
            (input = (m_in1 = PointMass(2.0), m_in2 = NormalWeightedMeanPrecision(2.0, 0.5), meta = NoCorrection()), output = NormalMeanVariance(8.0, 8.0))
        ]

        @test_rules [check_type_promotion = true] typeof(dot)(:out, Marginalisation) [
            (input = (m_in1 = PointMass(1.0), m_in2 = NormalMeanVariance(2.0, 2.0), meta = ReplaceZeroDiagonalEntries(tiny)), output = NormalMeanVariance(2.0, 2.0)),
            (input = (m_in1 = PointMass(-1.0), m_in2 = NormalMeanPrecision(3.0, 1.0), meta = ReplaceZeroDiagonalEntries(tiny)), output = NormalMeanVariance(-3.0, 1.0)),
            (input = (m_in1 = PointMass(2.0), m_in2 = NormalWeightedMeanPrecision(2.0, 0.5), meta = ReplaceZeroDiagonalEntries(tiny)), output = NormalMeanVariance(8.0, 8.0))
        ]
    end

    @testset "Belief Propagation: (m_in1::NormalDistributionsFamily, m_in2::PointMass)" begin
        @test_rules [check_type_promotion = true] typeof(dot)(:out, Marginalisation) [
            (input = (m_in1 = NormalMeanVariance(2.0, 2.0), m_in2 = PointMass(4.0), meta = NoCorrection()), output = NormalMeanVariance(8.0, 32.0)),
            (input = (m_in1 = NormalMeanPrecision(2.0, inv(3.0)), m_in2 = PointMass(2.0), meta = NoCorrection()), output = NormalMeanVariance(4.0, 12.0)),
            (input = (m_in1 = NormalWeightedMeanPrecision(2.0, 0.5), m_in2 = PointMass(1.0), meta = NoCorrection()), output = NormalMeanVariance(4.0, 2.0))
        ]

        @test_rules [check_type_promotion = true] typeof(dot)(:out, Marginalisation) [
            (input = (m_in1 = NormalMeanVariance(2.0, 2.0), m_in2 = PointMass(4.0), meta = ReplaceZeroDiagonalEntries(tiny)), output = NormalMeanVariance(8.0, 32.0)),
            (input = (m_in1 = NormalMeanPrecision(2.0, inv(3.0)), m_in2 = PointMass(2.0), meta = ReplaceZeroDiagonalEntries(tiny)), output = NormalMeanVariance(4.0, 12.0)),
            (input = (m_in1 = NormalWeightedMeanPrecision(2.0, 0.5), m_in2 = PointMass(1.0), meta = ReplaceZeroDiagonalEntries(tiny)), output = NormalMeanVariance(4.0, 2.0))
        ]
    end

    @testset "Belief Propagation: (m_in1::MultivariateNormalDistributionsFamily, m_in2::PointMass)" begin
        @test_rules [check_type_promotion = true] typeof(dot)(:out, Marginalisation) [
            (
                input = (m_in1 = MvNormalMeanCovariance([-1.0, 1.0], [2.0 -1.0; -1.0 4.0]), m_in2 = PointMass([4.0, 1.0]), meta = ReplaceZeroDiagonalEntries(tiny)),
                output = NormalMeanVariance(-3, 28)
            ),
            (
                input = (m_in1 = MvNormalMeanPrecision([2.0, 1.0], [2.0 -0.5; -0.5 5.0]), m_in2 = PointMass([2.0, 2.0]), meta = ReplaceZeroDiagonalEntries(tiny)),
                output = NormalMeanVariance(6.0, 128 / 39)
            ),
            (
                input = (m_in1 = MvNormalWeightedMeanPrecision([3.0, 2.0], [10.0 1.0; 1.0 20.0]), m_in2 = PointMass([-1.0, 3.0]), meta = ReplaceZeroDiagonalEntries(tiny)),
                output = NormalMeanVariance(-7 / 199, 116 / 199)
            )
        ]

        @test_rules [check_type_promotion = true] typeof(dot)(:out, Marginalisation) [
            (
                input = (m_in1 = MvNormalMeanCovariance([-1.0, 1.0], [2.0 -1.0; -1.0 4.0]), m_in2 = PointMass([4.0, 1.0]), meta = NoCorrection()),
                output = NormalMeanVariance(-3, 28)
            ),
            (
                input = (m_in1 = MvNormalMeanPrecision([2.0, 1.0], [2.0 -0.5; -0.5 5.0]), m_in2 = PointMass([2.0, 2.0]), meta = NoCorrection()),
                output = NormalMeanVariance(6.0, 128 / 39)
            ),
            (
                input = (m_in1 = MvNormalWeightedMeanPrecision([3.0, 2.0], [10.0 1.0; 1.0 20.0]), m_in2 = PointMass([-1.0, 3.0]), meta = NoCorrection()),
                output = NormalMeanVariance(-7 / 199, 116 / 199)
            )
        ]
    end

    @testset "Belief Propagation: (m_in1::PointMass, m_in2::MultivariateNormalDistributionsFamily)" begin
        @test_rules [check_type_promotion = true] typeof(dot)(:out, Marginalisation) [
            (
                input = (m_in1 = PointMass([4.0, 1.0]), m_in2 = MvNormalMeanCovariance([-1.0, 1.0], [2.0 -1.0; -1.0 4.0]), meta = ReplaceZeroDiagonalEntries(tiny)),
                output = NormalMeanVariance(-3, 28)
            ),
            (
                input = (m_in1 = PointMass([2.0, 2.0]), m_in2 = MvNormalMeanPrecision([2.0, 1.0], [2.0 -0.5; -0.5 5.0]), meta = ReplaceZeroDiagonalEntries(tiny)),
                output = NormalMeanVariance(6.0, 128 / 39)
            ),
            (
                input = (m_in1 = PointMass([-1.0, 3.0]), m_in2 = MvNormalWeightedMeanPrecision([3.0, 2.0], [10.0 1.0; 1.0 20.0]), meta = ReplaceZeroDiagonalEntries(tiny)),
                output = NormalMeanVariance(-7 / 199, 116 / 199)
            )
        ]

        @test_rules [check_type_promotion = true] typeof(dot)(:out, Marginalisation) [
            (
                input = (m_in1 = PointMass([4.0, 1.0]), m_in2 = MvNormalMeanCovariance([-1.0, 1.0], [2.0 -1.0; -1.0 4.0]), meta = NoCorrection()),
                output = NormalMeanVariance(-3, 28)
            ),
            (
                input = (m_in1 = PointMass([2.0, 2.0]), m_in2 = MvNormalMeanPrecision([2.0, 1.0], [2.0 -0.5; -0.5 5.0]), meta = NoCorrection()),
                output = NormalMeanVariance(6.0, 128 / 39)
            ),
            (
                input = (m_in1 = PointMass([-1.0, 3.0]), m_in2 = MvNormalWeightedMeanPrecision([3.0, 2.0], [10.0 1.0; 1.0 20.0]), meta = NoCorrection()),
                output = NormalMeanVariance(-7 / 199, 116 / 199)
            )
        ]
    end

    @testset "Error belief Propagation: (m_in1::NormalDistributionsFamily, m_in2::NormalDistributionsFamily)" begin
        @test_throws r"The rule for the dot product node between two NormalDistributionsFamily instances is not available in closed form. Please use SoftDot instead." @call_rule typeof(
            dot
        )(
            :out, Marginalisation
        ) (m_in1 = NormalMeanVariance(2.0, 2.0), m_in2 = NormalMeanVariance(2.0, 2.0), meta = NoCorrection())
    end
end
