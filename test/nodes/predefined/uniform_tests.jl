@testitem "UniformNode" begin
    using ReactiveMP, Random, BayesBase, ExponentialFamily, Distributions

    @testset "Average energy" begin
        a, b = 0.0, 1.0
        α, β = rand(0.1:0.1:1.0), rand(0.1:0.1:1.0)
        @test score(
            AverageEnergy(),
            Uniform,
            Val{(:out, :a, :b)}(),
            (Marginal(Beta(α, β), false, false, nothing), Marginal(PointMass(a), false, false, nothing), Marginal(PointMass(b), false, false, nothing)),
            nothing
        ) == 0.0
    end

    @testset "Product rule" begin
        # The default product rule between Uniform and Beta
        @test BayesBase.default_prod_rule(Uniform, Beta) == PreserveTypeProd(Distribution)

        # The special case for Uniform(0, 1)
        u = Uniform(0.0, 1.0)
        b = Beta(2.0, 5.0)

        result = prod(PreserveTypeProd(Distribution), u, b)

        @test result == b              # Uniform(0,1) leaves Beta unchanged
        @test result isa Beta          # Check the type
    end

    @testset "Node registration" begin
        # Basic sanity check for the node definition
        @test ReactiveMP.interfaces(Uniform) == Val((:out, :a, :b))
    end
end
