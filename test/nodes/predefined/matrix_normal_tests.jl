
@testitem "MatrixNormalNode" begin
    using ReactiveMP, Random, BayesBase, ExponentialFamily, Distributions, LinearAlgebra
    using FastCholesky

    import ReactiveMP: alias_interface

    @testset "Node is registered" begin
        @test ReactiveMP.is_predefined_node(MatrixNormal) isa
            ReactiveMP.PredefinedNodeFunctionalForm
        @test ReactiveMP.sdtype(MatrixNormal) === ReactiveMP.Stochastic()
        @test ReactiveMP.interfaces(MatrixNormal) === Val((:out, :M, :U, :V))
    end

    @testset "Interface aliases" begin
        @test alias_interface(MatrixNormal, 2, :mean) === :M
        @test alias_interface(MatrixNormal, 3, :rowcov) === :U
        @test alias_interface(MatrixNormal, 4, :colcov) === :V
    end

    @testset "AverageEnergy" begin
        score_ae(marginals) = score(
            AverageEnergy(),
            MatrixNormal,
            Val{(:out, :M, :U, :V)}(),
            marginals,
            nothing,
        )

        # All point masses: the average energy reduces to -logpdf.
        for (X, M, U, V) in (
            (
                [1.0 2.0; 3.0 4.0],
                [0.5 1.0; 1.5 2.0],
                [2.0 0.3; 0.3 1.5],
                [1.0 0.2; 0.2 2.0],
            ),
            (
                [1.0 2.0; 3.0 4.0; 5.0 6.0],
                [0.5 1.0; 2.0 3.0; 4.0 5.0],
                [2.0 0.0 0.0; 0.0 3.0 0.0; 0.0 0.0 4.0],
                [1.0 0.5; 0.5 2.0],
            ),
        )
            marginals = (
                Marginal(PointMass(X), false, false),
                Marginal(PointMass(M), false, false),
                Marginal(PointMass(U), false, false),
                Marginal(PointMass(V), false, false),
            )
            @test score_ae(marginals) ≈ -logpdf(MatrixNormal(M, U, V), X)
        end

        # q_out::MatrixNormal contributes a second-moment term tr(V⁻¹ V_out) U_out.
        begin
            X = [1.0 2.0; 3.0 4.0]
            U_out = [0.5 0.0; 0.0 0.5]
            V_out = [1.0 0.0; 0.0 0.5]
            M = [0.5 1.0; 1.5 2.0]
            U = [2.0 0.3; 0.3 1.5]
            V = [1.0 0.2; 0.2 2.0]
            marginals = (
                Marginal(MatrixNormal(X, U_out, V_out), false, false),
                Marginal(PointMass(M), false, false),
                Marginal(PointMass(U), false, false),
                Marginal(PointMass(V), false, false),
            )
            n, p = size(X)
            invU, invV = inv(U), inv(V)
            D = X - M
            Ψ = D * invV * D' + tr(invV * V_out) * U_out
            expected =
                (p * logdet(U) + n * logdet(V) + n * p * log(2π) + tr(invU * Ψ)) / 2
            @test score_ae(marginals) ≈ expected
        end

        # InverseWishart marginals on U and V flow through E[log|·|] and E[·⁻¹].
        begin
            X = [1.0 2.0; 3.0 4.0]
            M = [0.5 1.0; 1.5 2.0]
            q_U = InverseWishart(6.0, [2.0 0.0; 0.0 2.0])
            q_V = InverseWishart(5.0, [1.0 0.0; 0.0 1.0])
            marginals = (
                Marginal(PointMass(X), false, false),
                Marginal(PointMass(M), false, false),
                Marginal(q_U, false, false),
                Marginal(q_V, false, false),
            )
            n, p = size(X)
            invU = mean(cholinv, q_U)
            invV = mean(cholinv, q_V)
            D = X - M
            expected =
                (
                    p * mean(logdet, q_U) +
                    n * mean(logdet, q_V) +
                    n * p * log(2π) +
                    tr(invU * (D * invV * D'))
                ) / 2
            @test score_ae(marginals) ≈ expected
        end
    end
end
