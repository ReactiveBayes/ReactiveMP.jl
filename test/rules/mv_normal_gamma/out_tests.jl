
@testitem "rules:MvNormalGamma:out" begin
    using ReactiveMP, BayesBase, ExponentialFamily, Distributions, LinearAlgebra

    import ReactiveMP: @call_rule

    @testset "PointMass hyperparameters pass through" begin
        order = 2
        μ0, Λ0, α0, β0 = [0.5, -1.0], [2.0 0.3; 0.3 1.5], 2.0, 3.0

        msg = @call_rule MvNormalGamma(:out, Marginalisation) (
            m_μ = PointMass(μ0), m_Λ = PointMass(Λ0), m_α = PointMass(α0), m_β = PointMass(β0)
        )
        μ, Λ, α, β = params(msg)
        @test msg isa MvNormalGamma
        @test μ ≈ μ0
        @test Λ ≈ Λ0
        @test α ≈ α0
        @test β ≈ β0
    end

    @testset "q_ (Any) variant uses means" begin
        order = 1
        msg = @call_rule MvNormalGamma(:out, Marginalisation) (
            q_μ = PointMass([0.0]), q_Λ = PointMass(fill(1.0, 1, 1)), q_α = PointMass(1.0), q_β = PointMass(1.0)
        )
        μ, Λ, α, β = params(msg)
        @test μ ≈ [0.0]
        @test Λ ≈ fill(1.0, 1, 1)
        @test α ≈ 1.0
        @test β ≈ 1.0
    end
end
