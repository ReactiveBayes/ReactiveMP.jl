module ReactiveMPTestingHelpers

using ReactiveMP

@testset "Helpers" begin
    
    @testset "OneDivNVector" begin 

        for type in [ Float64, Float32, BigFloat ], len in [ 3, 5, 10 ]
            iter = OneDivNVector(type, len)

            @test eltype(iter) === type
            @test length(iter) === len
            @test size(iter)   === (len, )
            @test collect(iter) == fill(one(type) / len, len)
            @test sizeof(iter) === 0

        end

        @test eltype(OneDivNVector(3)) === Float64
        @test eltype(OneDivNVector(5)) === Float64

        @test_throws AssertionError OneDivNVector(-2)
        @test_throws AssertionError OneDivNVector(-10)
        @test_throws AssertionError OneDivNVector(Vector, 2)
        @test_throws AssertionError OneDivNVector(Matrix, 10)

    end
    
end

end