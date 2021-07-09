module ProbitNodeTest

using Test
using ReactiveMP
using Random

@testset "ProbitNode" begin

    @testset "Creation" begin
        node = make_node(Probit)

        @test functionalform(node)          === Probit
        @test sdtype(node)                  === Stochastic()
        @test name.(interfaces(node))       === (:out, :in)
        @test factorisation(node)           === ((1, 2), )
        @test metadata(node)                === nothing

        node = make_node(Probit, meta = 1)

        @test functionalform(node)          === Probit
        @test sdtype(node)                  === Stochastic()
        @test name.(interfaces(node))       === (:out, :in)
        @test factorisation(node)           === ((1, 2), )
        @test metadata(node)                === 1
    end

end

end
