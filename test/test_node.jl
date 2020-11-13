module ReactiveMPNodeTest

using Test
using ReactiveMP 

@testset "Node" begin

    @testset begin
        @test isdeterministic(Deterministic()) === true
        @test isdeterministic(Deterministic)   === true
        @test isdeterministic(Stochastic())    === false
        @test isdeterministic(Stochastic)      === false
        @test isstochastic(Deterministic())    === false
        @test isstochastic(Deterministic)      === false
        @test isstochastic(Stochastic())       === true
        @test isstochastic(Stochastic)         === true
    end
    
end

end