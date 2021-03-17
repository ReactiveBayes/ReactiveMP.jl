module ReactiveMPMessageTest

using Test
using ReactiveMP 

import Base.Iterators: repeated, product

@testset "Message" begin

    @testset "Constructor" begin 
        data    = PointMass(1)

        @test getdata(Message(data, true, true))    === data
        @test is_clamped(Message(data, true, true)) === true
        @test is_initial(Message(data, true, true)) === true

        @test getdata(Message(data, true, false))    === data
        @test is_clamped(Message(data, true, false)) === true
        @test is_initial(Message(data, true, false)) === false

        @test getdata(Message(data, false, true))    === data
        @test is_clamped(Message(data, false, true)) === false
        @test is_initial(Message(data, false, true)) === true

        @test getdata(Message(data, false, false))    === data
        @test is_clamped(Message(data, false, false)) === false
        @test is_initial(Message(data, false, false)) === false
    end
    
    @testset "multiply_messages" begin 
        dist1 = NormalMeanVariance(randn(), rand())
        dist2 = NormalMeanVariance(randn(), rand())

        @test getdata(Message(dist1, false, false) * Message(dist2, false, false))    == prod(ProdPreserveParametrisation(), dist1, dist2)
        @test getdata(Message(dist2, false, false) * Message(dist1, false, false))    == prod(ProdPreserveParametrisation(), dist2, dist1)

        for (left_is_initial, right_is_initial) in product(repeated([ true, false ], 2)...)
            @test is_clamped(Message(dist1, true, left_is_initial) * Message(dist2, false, right_is_initial)) == false
            @test is_clamped(Message(dist1, false, left_is_initial) * Message(dist2, true, right_is_initial)) == false
            @test is_clamped(Message(dist1, true, left_is_initial) * Message(dist2, true, right_is_initial))  == true
            @test is_clamped(Message(dist2, true, left_is_initial) * Message(dist1, false, right_is_initial)) == false
            @test is_clamped(Message(dist2, false, left_is_initial) * Message(dist1, true, right_is_initial)) == false
            @test is_clamped(Message(dist2, true, left_is_initial) * Message(dist1, true, right_is_initial))  == true
        end

        for (left_is_clamped, right_is_clamped) in product(repeated([ true, false ], 2)...)
            @test is_initial(Message(dist1, left_is_clamped, true) * Message(dist2, right_is_clamped, true)) == !(left_is_clamped && right_is_clamped)
            @test is_initial(Message(dist2, left_is_clamped, true) * Message(dist1, right_is_clamped, true)) == !(left_is_clamped && right_is_clamped)
            @test is_initial(Message(dist1, left_is_clamped, false) * Message(dist2, right_is_clamped, false)) == false
            @test is_initial(Message(dist2, left_is_clamped, false) * Message(dist1, right_is_clamped, false)) == false    
        end

        @test is_initial(Message(dist1, true, true) * Message(dist2, true, true)) == false
        @test is_initial(Message(dist1, true, true) * Message(dist2, true, false)) == false
        @test is_initial(Message(dist1, true, false) * Message(dist2, true, true)) == false
        @test is_initial(Message(dist1, false, true) * Message(dist2, true, false)) == true
        @test is_initial(Message(dist1, true, false) * Message(dist2, false, true)) == true
        @test is_initial(Message(dist2, true, true) * Message(dist1, true, true)) == false
        @test is_initial(Message(dist2, true, true) * Message(dist1, true, false)) == false
        @test is_initial(Message(dist2, true, false) * Message(dist1, true, true)) == false
        @test is_initial(Message(dist2, false, true) * Message(dist1, true, false)) == true
        @test is_initial(Message(dist2, true, false) * Message(dist1, false, true)) == true
    end

end

end