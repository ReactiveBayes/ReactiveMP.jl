@testitem "marginalrules:Multiplication" begin
    using ReactiveMP, BayesBase, Random, ExponentialFamily, Distributions

    import ReactiveMP: @test_marginalrules

    @testset "out_A_in: (m_out::NormalMeanPrecision, m_A::PointMass, m_in::NormalMeanPrecision)" begin
        @test_marginalrules [check_type_promotion = false] (*)(:A_in) [
            (
                input = (m_out = NormalMeanPrecision(0.0, 1.0), m_A = PointMass(1.0), m_in = NormalMeanPrecision(1.0, 2.0)), 
                output = (A = PointMass(1.0), in = NormalWeightedMeanPrecision(2.0, 3.0))
            ),

            (
                input = (m_out = NormalMeanPrecision(1.0, 2.0), m_A = PointMass(1.0), m_in = NormalMeanPrecision(2.0, 1.0)), 
                output = (A = PointMass(1.0), in = NormalWeightedMeanPrecision(4.0, 3.0))
            ),

            (
                input = (m_out = NormalMeanPrecision(0.0, 2.0), m_A = PointMass(2.0), m_in = NormalMeanPrecision(1.0, 1.0)), 
                output = (A = PointMass(2.0), in = NormalWeightedMeanPrecision(1.0, 9.0))
            ),
        ]
    end

    @testset "out_A_in: (m_out::NormalMeanPrecision, m_A::NormalMeanPrecision, m_in::PointMass)" begin
        @test_marginalrules [check_type_promotion = false] (*)(:A_in) [
            (
                input = (m_out = NormalMeanPrecision(0.0, 1.0), m_A = NormalMeanPrecision(1.0, 2.0), m_in = PointMass(1.0)), 
                output = (A = NormalWeightedMeanPrecision(2.0, 3.0), in = PointMass(1.0))
            ),

            (
                input = (m_out = NormalMeanPrecision(1.0, 2.0), m_A = NormalMeanPrecision(2.0, 1.0), m_in = PointMass(1.0)), 
                output = (A = NormalWeightedMeanPrecision(4.0, 3.0), in = PointMass(1.0))
            ),

            (
                input = (m_out = NormalMeanPrecision(0.0, 2.0), m_A = NormalMeanPrecision(1.0, 1.0), m_in = PointMass(2.0)), 
                output = (A = NormalWeightedMeanPrecision(1.0, 9.0), in = PointMass(2.0))
            ),
        ]
    end

    @testset "out_A_in: (m_out::MvNormalMeanPrecision, m_A::NormalMeanPrecision, m_in::PointMass)" begin
        @test_marginalrules [check_type_promotion = false] (*)(:A_in) [
            (
                input = (m_out = MvNormalMeanPrecision([0.0], [1.0;;]), m_A = NormalMeanPrecision(0.0, 1.0), m_in = PointMass([1.0])),
                output = (A = NormalWeightedMeanPrecision(0.0, 2.0), in = PointMass([1.0]))
            ),

            (
                input = (m_out = MvNormalMeanPrecision([1.0], [2.0;;]), m_A = NormalMeanPrecision(2.0, 1.0), m_in = PointMass([0.0])),
                output = (A = NormalWeightedMeanPrecision(2.0, 1.0), in = PointMass([0.0]))
            ),

            (
                input = (m_out = MvNormalMeanPrecision([2.0], [1.0;;]), m_A = NormalMeanPrecision(1.0, 2.0), m_in = PointMass([2.0])),
                output = (A = NormalWeightedMeanPrecision(6.0, 6.0), in = PointMass([2.0]))
            ),
        ]
    end
end
