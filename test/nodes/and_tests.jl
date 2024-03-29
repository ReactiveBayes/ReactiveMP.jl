
@testitem "AndNode" begin
    using ReactiveMP, Random
    @testset "Creation" begin
        node = make_node(AND)

        @test functionalform(node) === AND
        @test sdtype(node) === Deterministic()
        @test name.(interfaces(node)) === (:out, :in1, :in2)
        @test factorisation(node) === ((1, 2, 3),)
        @test localmarginalnames(node) === (:out_in1_in2,)
        @test metadata(node) === nothing
    end
end
