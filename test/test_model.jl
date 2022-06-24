module ReactiveMPModelTest

using Test
using ReactiveMP
using GraphPPL
using Random

# TODO: move this tests to RxInfer.jl
@testset "@model macro tests" begin
    @testset "Broadcasting syntax #1" begin 

        @model function bsyntax1(n, broadcasting)
            m ~ NormalMeanPrecision(0.0, 1.0)
            t ~ Gamma(1.0, 1.0)
            y = datavar(Float64, n)

            nodes = Vector{Any}(undef, n)

            if broadcasting
                nodes, y .~ NormalMeanPrecision(m, t)
            else 
                for i in 1:n
                    nodes[i], y[i] ~ NormalMeanPrecision(m, t)
                end
            end

            return nodes, y, m, t
        end

        n = 10
        # Test that both model create without any issues
        modelb, (nodesb, yb, mb, tb) = bsyntax1(n, true)
        model, (nodes, y, m, t)      = bsyntax1(n, false)

        # Test that degrees match
        @test ReactiveMP.degree(mb) === n + 1
        @test ReactiveMP.degree(m) === n + 1
        @test ReactiveMP.degree(tb) === n + 1
        @test ReactiveMP.degree(t) === n + 1

        for testset in (((nodesb, yb), mb, tb), ((nodes, y), m, t))
            for (node, y) in zip(testset[1]...) 
                @test ReactiveMP.connectedvar(ReactiveMP.getinterface(node, :out)) === y # Test that :out interface has been connected to `y[i]`
                @test ReactiveMP.connectedvar(ReactiveMP.getinterface(node, :μ)) === testset[2] # Test that :μ interface has been connected to 'm'
                @test ReactiveMP.connectedvar(ReactiveMP.getinterface(node, :τ)) === testset[3] # Test that :τ interface has been connected to 't'
            end
        end
    end

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
