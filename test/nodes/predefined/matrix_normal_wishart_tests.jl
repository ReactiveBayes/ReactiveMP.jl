
@testitem "MatrixNormalWishartNode" begin
    using ReactiveMP,
        Random, BayesBase, ExponentialFamily, Distributions, LinearAlgebra
    using FastCholesky, StableRNGs

    import ReactiveMP: alias_interface

    @testset "Node is registered" begin
        @test ReactiveMP.is_predefined_node(MatrixNormalWishart) isa
            ReactiveMP.PredefinedNodeFunctionalForm
        @test ReactiveMP.sdtype(MatrixNormalWishart) === ReactiveMP.Stochastic()
        @test ReactiveMP.interfaces(MatrixNormalWishart) ===
            Val((:out, :M, :U, :V, :ν))
    end

    @testset "Interface aliases" begin
        @test alias_interface(MatrixNormalWishart, 2, :mean) === :M
        @test alias_interface(MatrixNormalWishart, 3, :rowcov) === :U
        @test alias_interface(MatrixNormalWishart, 4, :scale) === :V
        @test alias_interface(MatrixNormalWishart, 5, :dof) === :ν
    end

    @testset "AverageEnergy (Monte Carlo)" begin
        # factor hyperparameters (point masses)
        M = [1.0 2.0; 3.0 4.0]
        U = [2.0 0.3; 0.3 1.5]
        V = [1.0 0.2; 0.2 1.5]
        ν = 5.0

        # joint belief over out = (X, Y), distinct from the hyperparameters so D = Mq - M ≠ 0
        q_out = MatrixNormalWishart(
            [0.5 1.0; 1.5 2.0], [1.5 0.2; 0.2 1.0], [1.0 0.1; 0.1 1.2], 6.0
        )

        marginals = (
            Marginal(q_out, false, false),
            Marginal(PointMass(M), false, false),
            Marginal(PointMass(U), false, false),
            Marginal(PointMass(V), false, false),
            Marginal(PointMass(ν), false, false),
        )

        analytic = score(
            AverageEnergy(),
            MatrixNormalWishart,
            Val{(:out, :M, :U, :V, :ν)}(),
            marginals,
            nothing,
        )

        # Monte-Carlo estimate of E_{q_out}[-log f], with the factor density written out as
        # the matrix-normal / Wishart product (avoids pdf -> log underflow).
        # Keep N modest to bound CI time; the fixed StableRNG seed makes the
        # estimate deterministic, and rtol is loosened to absorb the extra
        # variance from fewer samples.
        rng = StableRNG(42)
        N = 5_000
        acc = 0.0
        for _ in 1:N
            X, Y = rand(rng, q_out)
            acc -=
                logpdf(MatrixNormal(M, U, cholinv(Y)), X) +
                logpdf(Wishart(ν, V), Y)
        end
        mc = acc / N

        @test analytic ≈ mc rtol = 0.1
    end
end
