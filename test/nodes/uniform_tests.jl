
@testitem "UniformNode" begin
    using ReactiveMP, Random, BayesBase, ExponentialFamily, Distributions

    import ReactiveMP: make_node

    @testset "Creation" begin
        node = make_node(Uniform)

        @test functionalform(node) === Uniform
        @test sdtype(node) === Stochastic()
        @test name.(interfaces(node)) === (:out, :a, :b)
        @test factorisation(node) === ((1, 2, 3),)
        @test localmarginalnames(node) === (:out_a_b,)
        @test metadata(node) === nothing

        node = make_node(Uniform, FactorNodeCreationOptions(nothing, 1, nothing))

        @test metadata(node) === 1
    end

    @testset "Average energy" begin
        node = ReactiveMP.make_node(Uniform)
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
end
