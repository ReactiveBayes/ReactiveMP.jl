module ReactiveMPVariableTest

using Test
using ReactiveMP
using Rocket

@testset "Variable" begin
    @testset "setmarginal! tests for randomvar" begin
        

        for dist in (NormalMeanVariance(-2.0, 3.0), NormalMeanPrecision(-2.0, 3.0))

            T = typeof(dist)
            variable = randomvar(:r)
            flag = false

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

            variables = randomvar(:r, 2)
            flagmv = false

            setmarginals!(variables, dist)

            subscriptionmv = subscribe!(
                getmarginals(variables, IncludeAll()),
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
