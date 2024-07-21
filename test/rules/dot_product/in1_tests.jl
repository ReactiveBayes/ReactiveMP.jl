
@testitem "rules:typeof(dot):in1" begin
    using ReactiveMP, BayesBase, Random, ExponentialFamily, Distributions

    import ReactiveMP: @test_rules
    import LinearAlgebra: dot
    import MatrixCorrectionTools: NoCorrection, AddToDiagonalEntries, ReplaceZeroDiagonalEntries

    @testset "Belief Propagation: (m_out::UnivariateNormalDistributionsFamily, m_in2::PointMass)" begin
        @test_rules [check_type_promotion = true] typeof(dot)(:in1, Marginalisation) [
            (input = (m_out = NormalMeanVariance(2.0, 2.0), m_in2 = PointMass(-1.0), meta = NoCorrection()), output = NormalWeightedMeanPrecision(-1.0, 0.5)),
            (input = (m_out = NormalMeanPrecision(1.0, 1.0), m_in2 = PointMass(-2.0), meta = NoCorrection()), output = NormalWeightedMeanPrecision(-2.0, 4.0)),
            (input = (m_out = NormalWeightedMeanPrecision(2.0, 1.0), m_in2 = PointMass(-1.0), meta = NoCorrection()), output = NormalWeightedMeanPrecision(-2.0, 1.0))
        ]

        @test_rules [check_type_promotion = true] typeof(dot)(:in1, Marginalisation) [
            (input = (m_out = NormalMeanVariance(2.0, 2.0), m_in2 = PointMass(-1.0), meta = ReplaceZeroDiagonalEntries(tiny)), output = NormalWeightedMeanPrecision(-1.0, 0.5)),
            (input = (m_out = NormalMeanPrecision(1.0, 1.0), m_in2 = PointMass(-2.0), meta = ReplaceZeroDiagonalEntries(tiny)), output = NormalWeightedMeanPrecision(-2.0, 4.0)),
            (
                input = (m_out = NormalWeightedMeanPrecision(2.0, 1.0), m_in2 = PointMass(-1.0), meta = ReplaceZeroDiagonalEntries(tiny)),
                output = NormalWeightedMeanPrecision(-2.0, 1.0)
            )
        ]

        @test_rules [check_type_promotion = true] typeof(dot)(:in1, Marginalisation) [
            (input = (m_out = NormalMeanVariance(2.0, 2.0), m_in2 = PointMass(0.0), meta = NoCorrection()), output = NormalWeightedMeanPrecision(0.0, 0.0)),
            (input = (m_out = NormalMeanPrecision(1.0, 1.0), m_in2 = PointMass(0.0), meta = NoCorrection()), output = NormalWeightedMeanPrecision(0.0, 0.0)),
            (input = (m_out = NormalWeightedMeanPrecision(2.0, 1.0), m_in2 = PointMass(0.0), meta = NoCorrection()), output = NormalWeightedMeanPrecision(0.0, 0.0))
        ]

        @test_rules [check_type_promotion = true] typeof(dot)(:in1, Marginalisation) [
            (input = (m_out = NormalMeanVariance(2.0, 2.0), m_in2 = PointMass(0.0), meta = ReplaceZeroDiagonalEntries(tiny)), output = NormalWeightedMeanPrecision(0.0, tiny)),
            (input = (m_out = NormalMeanPrecision(1.0, 1.0), m_in2 = PointMass(0.0), meta = ReplaceZeroDiagonalEntries(tiny)), output = NormalWeightedMeanPrecision(0.0, tiny)),
            (
                input = (m_out = NormalWeightedMeanPrecision(2.0, 1.0), m_in2 = PointMass(0.0), meta = ReplaceZeroDiagonalEntries(tiny)),
                output = NormalWeightedMeanPrecision(0.0, tiny)
            )
        ]
    end

    @testset "Belief Propagation: (m_out::UnivariateNormalDistributionsFamily, m_in2::PointMass)" begin
        @test_rules [check_type_promotion = true] typeof(dot)(:in1, Marginalisation) [
            (
                input = (m_out = NormalMeanVariance(2.0, 1.0), m_in2 = PointMass([-1.0, 2.0]), meta = NoCorrection()),
                output = MvNormalWeightedMeanPrecision([-2.0, 4.0], [1.0 -2.0; -2.0 4.0])
            ),
            (
                input = (m_out = NormalMeanPrecision(1.0, inv(2.0)), m_in2 = PointMass([1.0, 1.0]), meta = NoCorrection()),
                output = MvNormalWeightedMeanPrecision([0.5, 0.5], [0.5 0.5; 0.5 0.5])
            ),
            (
                input = (m_out = NormalWeightedMeanPrecision(2.0, 1.0), m_in2 = PointMass([-2.0, 3.0]), meta = NoCorrection()),
                output = MvNormalWeightedMeanPrecision([-4.0, 6.0], [4.0 -6.0; -6.0 9.0])
            )
        ]

        @test_rules [check_type_promotion = true] typeof(dot)(:in1, Marginalisation) [
            (
                input = (m_out = NormalMeanVariance(2.0, 1.0), m_in2 = PointMass([-1.0, 2.0]), meta = ReplaceZeroDiagonalEntries(tiny)),
                output = MvNormalWeightedMeanPrecision([-2.0, 4.0], [1.0 -2.0; -2.0 4.0])
            ),
            (
                input = (m_out = NormalMeanPrecision(1.0, inv(2.0)), m_in2 = PointMass([1.0, 1.0]), meta = ReplaceZeroDiagonalEntries(tiny)),
                output = MvNormalWeightedMeanPrecision([0.5, 0.5], [0.5 0.5; 0.5 0.5])
            ),
            (
                input = (m_out = NormalWeightedMeanPrecision(2.0, 1.0), m_in2 = PointMass([-2.0, 3.0]), meta = ReplaceZeroDiagonalEntries(tiny)),
                output = MvNormalWeightedMeanPrecision([-4.0, 6.0], [4.0 -6.0; -6.0 9.0])
            )
        ]

        @test_rules [check_type_promotion = true] typeof(dot)(:in1, Marginalisation) [
            (
                input = (m_out = NormalMeanVariance(2.0, 1.0), m_in2 = PointMass([-1.0, 2.0]), meta = AddToDiagonalEntries(tiny)),
                output = MvNormalWeightedMeanPrecision([-2.0, 4.0], [1.0+tiny -2.0; -2.0 4.0+tiny])
            ),
            (
                input = (m_out = NormalMeanPrecision(1.0, inv(2.0)), m_in2 = PointMass([1.0, 1.0]), meta = AddToDiagonalEntries(tiny)),
                output = MvNormalWeightedMeanPrecision([0.5, 0.5], [0.5+tiny 0.5; 0.5 0.5+tiny])
            ),
            (
                input = (m_out = NormalWeightedMeanPrecision(2.0, 1.0), m_in2 = PointMass([-2.0, 3.0]), meta = AddToDiagonalEntries(tiny)),
                output = MvNormalWeightedMeanPrecision([-4.0, 6.0], [4.0+tiny -6.0; -6.0 9.0+tiny])
            )
        ]
    end

    @testset "Error Belief Propagation: (m_out::UnivariateNormalDistributionsFamily, m_in2::NormalDistributionsFamily)" begin
        @test_throws r"The rule for the dot product node between two NormalDistributionsFamily instances is not available in closed form. Please use SoftDot instead." @call_rule typeof(
            dot
        )(
            :in1, Marginalisation
        ) (m_out = NormalMeanVariance(2.0, 2.0), m_in2 = MvNormalMeanCovariance([-1.0, 1.0], [2.0 -1.0; -1.0 4.0]), meta = NoCorrection())
    end
end
