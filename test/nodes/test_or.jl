module OrNodeTest

using ReactiveMP, Random, BayesBase, ExponentialFamily

@testitem "OrNode" begin
    @testset "Creation" begin
        node = make_node(OR)

        @test functionalform(node) === OR
        @test sdtype(node) === Deterministic()
        @test name.(interfaces(node)) === (:out, :in1, :in2)
        @test factorisation(node) === ((1, 2, 3),)
        @test localmarginalnames(node) === (:out_in1_in2,)
        @test metadata(node) === nothing
    end
end
end
