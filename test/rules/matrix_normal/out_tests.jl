
@testitem "rules:MatrixNormal:out" begin
    using ReactiveMP,
        BayesBase, Random, ExponentialFamily, Distributions, LinearAlgebra

    import ReactiveMP: @test_rules

    @testset "Belief Propagation: (m_M::PointMass, m_U::PointMass, m_V::PointMass)" begin
        @test_rules [check_type_promotion = true] MatrixNormal(
            :out, Marginalisation
        ) [
            (
                input = (
                    m_M = PointMass([1.0 2.0; 3.0 4.0]),
                    m_U = PointMass([2.0 0.0; 0.0 3.0]),
                    m_V = PointMass([1.0 0.0; 0.0 4.0]),
                ),
                output = MatrixNormal(
                    [1.0 2.0; 3.0 4.0], [2.0 0.0; 0.0 3.0], [1.0 0.0; 0.0 4.0]
                ),
            ),
            (
                input = (
                    m_M = PointMass([0.5 1.0; 2.0 3.0; 4.0 5.0]),
                    m_U = PointMass([2.0 0.0 0.0; 0.0 3.0 0.0; 0.0 0.0 4.0]),
                    m_V = PointMass([1.0 0.0; 0.0 2.0]),
                ),
                output = MatrixNormal(
                    [0.5 1.0; 2.0 3.0; 4.0 5.0],
                    [2.0 0.0 0.0; 0.0 3.0 0.0; 0.0 0.0 4.0],
                    [1.0 0.0; 0.0 2.0],
                ),
            ),
            (
                input = (
                    m_M = PointMass(zeros(2, 3)),
                    m_U = PointMass([1.0 0.0; 0.0 1.0]),
                    m_V = PointMass([1.0 0.0 0.0; 0.0 1.0 0.0; 0.0 0.0 1.0]),
                ),
                output = MatrixNormal(
                    zeros(2, 3),
                    [1.0 0.0; 0.0 1.0],
                    [1.0 0.0 0.0; 0.0 1.0 0.0; 0.0 0.0 1.0],
                ),
            ),
        ]
    end

    @testset "Belief Propagation: (m_M::MatrixNormal, m_U::PointMass, m_V::PointMass)" begin
        @test_rules [check_type_promotion = true] MatrixNormal(
            :out, Marginalisation
        ) [
            (
                input = (
                    m_M = MatrixNormal(
                        [1.0 2.0; 3.0 4.0],
                        [0.5 0.0; 0.0 0.5],
                        [1.0 0.0; 0.0 4.0],
                    ),
                    m_U = PointMass([2.0 0.5; 0.5 3.0]),
                    m_V = PointMass([1.0 0.0; 0.0 2.0]),
                ),
                output = MvNormalMeanCovariance(
                    vec([1.0 2.0; 3.0 4.0]),
                    kron([1.0 0.0; 0.0 2.0], [2.0 0.5; 0.5 3.0]) +
                    kron([1.0 0.0; 0.0 4.0], [0.5 0.0; 0.0 0.5]),
                ),
            ),
            (
                input = (
                    m_M = MatrixNormal(
                        zeros(3, 2),
                        Matrix(1.0 * I, 3, 3),
                        Matrix(1.0 * I, 2, 2),
                    ),
                    m_U = PointMass([2.0 0.0 0.0; 0.0 3.0 0.0; 0.0 0.0 4.0]),
                    m_V = PointMass([1.0 0.0; 0.0 2.0]),
                ),
                output = MvNormalMeanCovariance(
                    vec(zeros(3, 2)),
                    kron(
                        [1.0 0.0; 0.0 2.0],
                        [2.0 0.0 0.0; 0.0 3.0 0.0; 0.0 0.0 4.0],
                    ) + kron(Matrix(1.0 * I, 2, 2), Matrix(1.0 * I, 3, 3)),
                ),
            ),
        ]
    end

    @testset "Belief Propagation: (m_M::PointMass, m_U::InverseWishart, m_V::PointMass)" begin
        @test_rules [check_type_promotion = true] MatrixNormal(
            :out, Marginalisation
        ) [
            (
                input = (
                    m_M = PointMass([1.0 2.0; 3.0 4.0]),
                    m_U = InverseWishart(5.0, [2.0 0.0; 0.0 2.0]),
                    m_V = PointMass([1.0 0.0; 0.0 2.0]),
                ),
                output = MatrixTDist(
                    4.0,
                    [1.0 2.0; 3.0 4.0],
                    [2.0 0.0; 0.0 2.0],
                    [1.0 0.0; 0.0 2.0],
                ),
            ),
            (
                input = (
                    m_M = PointMass(zeros(3, 2)),
                    m_U = InverseWishart(
                        7.0,
                        [2.0 0.0 0.0; 0.0 3.0 0.0; 0.0 0.0 4.0],
                    ),
                    m_V = PointMass([1.0 0.5; 0.5 2.0]),
                ),
                output = MatrixTDist(
                    5.0,
                    zeros(3, 2),
                    [2.0 0.0 0.0; 0.0 3.0 0.0; 0.0 0.0 4.0],
                    [1.0 0.5; 0.5 2.0],
                ),
            ),
        ]
    end

    @testset "Belief Propagation: (m_M::PointMass, m_U::PointMass, m_V::InverseWishart)" begin
        @test_rules [check_type_promotion = true] MatrixNormal(
            :out, Marginalisation
        ) [
            (
                input = (
                    m_M = PointMass([1.0 2.0; 3.0 4.0]),
                    m_U = PointMass([2.0 0.5; 0.5 3.0]),
                    m_V = InverseWishart(5.0, [1.0 0.0; 0.0 1.0]),
                ),
                output = MatrixTDist(
                    4.0,
                    [1.0 2.0; 3.0 4.0],
                    [2.0 0.5; 0.5 3.0],
                    [1.0 0.0; 0.0 1.0],
                ),
            ),
            (
                input = (
                    m_M = PointMass(zeros(3, 2)),
                    m_U = PointMass([2.0 0.0 0.0; 0.0 3.0 0.0; 0.0 0.0 4.0]),
                    m_V = InverseWishart(6.0, [1.0 0.5; 0.5 2.0]),
                ),
                output = MatrixTDist(
                    5.0,
                    zeros(3, 2),
                    [2.0 0.0 0.0; 0.0 3.0 0.0; 0.0 0.0 4.0],
                    [1.0 0.5; 0.5 2.0],
                ),
            ),
        ]
    end

    @testset "Mean-field VMP: (q_M::MatrixNormal, q_U::InverseWishart, q_V::InverseWishart)" begin
        @test_rules [check_type_promotion = true] MatrixNormal(
            :out, Marginalisation
        ) [
            (
                input = (
                    q_M = MatrixNormal(
                        [1.0 2.0; 3.0 4.0],
                        [2.0 0.5; 0.5 3.0],
                        [1.0 0.0; 0.0 2.0],
                    ),
                    q_U = InverseWishart(5.0, [2.0 0.0; 0.0 2.0]),
                    q_V = InverseWishart(4.0, [1.0 0.0; 0.0 1.0]),
                ),
                output = MatrixNormal(
                    [1.0 2.0; 3.0 4.0],
                    [2.0 / 5.0 0.0; 0.0 2.0 / 5.0],
                    [1.0 / 4.0 0.0; 0.0 1.0 / 4.0],
                ),
            ),
            (
                input = (
                    q_M = MatrixNormal(
                        ones(3, 2),
                        Matrix(1.0 * I, 3, 3),
                        Matrix(1.0 * I, 2, 2),
                    ),
                    q_U = InverseWishart(
                        7.0,
                        [2.0 0.0 0.0; 0.0 3.0 0.0; 0.0 0.0 4.0],
                    ),
                    q_V = InverseWishart(5.0, [1.0 0.5; 0.5 2.0]),
                ),
                output = MatrixNormal(
                    ones(3, 2),
                    [2.0 / 7.0 0.0 0.0; 0.0 3.0 / 7.0 0.0; 0.0 0.0 4.0 / 7.0],
                    [1.0 / 5.0 0.5 / 5.0; 0.5 / 5.0 2.0 / 5.0],
                ),
            ),
        ]
    end
    # @testset "Mean-field VMP: (q_M::MatrixNormal, q_U::Union{PointMass,InverseWishart}, q_V::Union{PointMass,InverseWishart})" begin
    #     @test_rules [check_type_promotion = true] MatrixNormal(
    #         :out, Marginalisation
    #     ) [
    #         (
    #             input = (
    #                 q_M = MatrixNormal(
    #                     ones(3, 2),
    #                     Matrix(1.0 * I, 3, 3),
    #                     Matrix(1.0 * I, 2, 2),
    #                 ),
    #                 q_U = InverseWishart(
    #                     7.0,
    #                     [2.0 0.0 0.0; 0.0 3.0 0.0; 0.0 0.0 4.0],
    #                 ),
    #                 q_V = PointMass([1.0 0.5; 0.5 2.0]),
    #             ),
    #             output = MatrixNormal(
    #                 ones(3, 2),
    #                 [2.0 / 7.0 0.0 0.0; 0.0 3.0 / 7.0 0.0; 0.0 0.0 4.0 / 7.0],
    #                 [1.0 0.5; 0.5 2.0],
    #             ),
    #         ),
    #     ]
    # end
end
