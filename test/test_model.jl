module ReactiveMPModelTest

using Test
using ReactiveMP
using GraphPPL
using Random

# TODO: move this tests to RxInfer.jl
@testset "@model macro tests" begin

    @testset "Fail if variables has been overwritten" begin

        @model function mymodel1(; condition)
            if condition === 0
                x = randomvar()
                x = randomvar()
                x ~ NormalMeanPrecision(0.0, 1.0)
            elseif condition === 1
                x ~ NormalMeanPrecision(0.0, 1.0)
                x = randomvar()
            elseif condition === 2
                x = randomvar()
                x_saved = x
                x ~ NormalMeanPrecision(0.0, 1.0)
                @test x_saved === x
            elseif condition === 3
                x ~ NormalMeanPrecision(0.0, 1.0)
            end

            y = datavar(Float64)
            y ~ NormalMeanPrecision(x, 1.0)
        end

        @test_throws UndefVarError mymodel1(condition = -1)
        @test_throws ErrorException mymodel1(condition = 0)
        @test_throws ErrorException mymodel1(condition = 1)

        m, _ = mymodel1(condition = 2)
        @test haskey(m, :x) && ReactiveMP.degree(m[:x]) === 2

        m, _ = mymodel1(condition = 3)
        @test haskey(m, :x) && ReactiveMP.degree(m[:x]) === 2

    end

end

end