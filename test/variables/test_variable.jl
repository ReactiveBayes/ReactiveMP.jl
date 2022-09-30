module ReactiveMPVariableTest

using Test
using ReactiveMP
using Rocket

@testset "Variable" begin
    import ReactiveMP: activate!

    @testset "setmarginal! tests for randomvar" begin
        for dist in (NormalMeanVariance(-2.0, 3.0), NormalMeanPrecision(-2.0, 3.0))
            T = typeof(dist)
            variable = randomvar(:r)
            flag = false

            activate!(variable)

            setmarginal!(variable, dist)

            subscription = subscribe!(
                getmarginal(variable, IncludeAll()),
                (marginal) -> begin
                    @test typeof(marginal) <: Marginal{T}
                    @test mean(marginal) === mean(dist)
                    @test var(marginal) === var(dist)
                    flag = true
                end
            )

            # Test that subscription happenend
            @test flag === true

            unsubscribe!(subscription)

            variablesmv = randomvar(:r, 2)
            flagmv = false

            activate!.(variablesmv)

            setmarginals!(variablesmv, dist)

            subscriptionmv = subscribe!(
                getmarginals(variablesmv, IncludeAll()),
                (marginals) -> begin
                    @test length(marginals) === 2
                    foreach(marginals) do marginal
                        @test typeof(marginal) <: Marginal{T}
                        @test mean(marginal) === mean(dist)
                        @test var(marginal) === var(dist)
                    end
                    flagmv = true
                end
            )

            # Test that subscription happenend
            @test flagmv === true

            unsubscribe!(subscriptionmv)

            variablesmx = randomvar(:r, 2, 2)
            flagmx = false

            activate!.(variablesmx)

            setmarginals!(variablesmx, dist)

            subscriptionmx = subscribe!(getmarginals(variablesmx, IncludeAll()), (marginals) -> begin
                @test length(marginals) === 4
                foreach(marginals) do marginal
                    @test typeof(marginal) <: Marginal{T}
                    @test mean(marginal) === mean(dist)
                    @test var(marginal) === var(dist)
                end
                flagmx = true
            end)

            # Test that subscription happenend
            @test flagmx === true

            unsubscribe!(subscriptionmx)
        end
    end
end

end
