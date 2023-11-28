module ReactiveMPVariableTest

using Test, ReactiveMP, Rocket, BayesBase, Distributions, ExponentialFamily

@testset "Variable" begin
    import ReactiveMP: activate!
    import Rocket: getscheduler

    struct TestOptions end

    Rocket.getscheduler(::TestOptions) = AsapScheduler()
    Base.broadcastable(::TestOptions) = Ref(TestOptions()) # for broadcasting

    @testset "setmarginal! tests for randomvar" begin
        for dist in (NormalMeanVariance(-2.0, 3.0), NormalMeanPrecision(-2.0, 3.0), PointMass(2.0))
            T = typeof(dist)
            variable = randomvar(:r)
            flag = false

            activate!(variable, TestOptions())

            setmarginal!(variable, dist)

            subscription = subscribe!(getmarginal(variable, IncludeAll()), (marginal) -> begin
                @test typeof(marginal) <: Marginal{T}
                @test mean(marginal) === mean(dist)
                @test var(marginal) === var(dist)
                flag = true
            end)

            # Test that subscription happenend
            @test flag === true

            unsubscribe!(subscription)

            variablesmv = randomvar(:r, 2)
            flagmv = false

            activate!.(variablesmv, TestOptions())

            setmarginals!(variablesmv, dist)
            @test_throws AssertionError setmarginals!(variablesmv, dist[begin:end-1])

            subscriptionmv = subscribe!(getmarginals(variablesmv, IncludeAll()), (marginals) -> begin
                @test length(marginals) === 2
                foreach(marginals) do marginal
                    @test typeof(marginal) <: Marginal{T}
                    @test mean(marginal) === mean(dist)
                    @test var(marginal) === var(dist)
                end
                flagmv = true
            end)

            # Test that subscription happenend
            @test flagmv === true

            unsubscribe!(subscriptionmv)

            variablesmx = randomvar(:r, 2, 2)
            flagmx = false

            activate!.(variablesmx, TestOptions())

            setmarginals!(variablesmx, dist)
            @test_throws AssertionError setmarginals!(variablesmx, [dist])

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
