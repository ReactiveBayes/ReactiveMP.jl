module ReactiveMPVariableTest

using Test
using ReactiveMP
using Rocket

@testset "Variable" begin

    @testset "setmarginal!" begin 
    
        begin 
            r = randomvar(:r)
            setmarginal!(r, NormalMeanVariance(-2.0, 3.0))
            subscribe!(getmarginal(r, IncludeAll()), (marginal) -> begin
                @test typeof(marginal) <: Marginal{ <: NormalMeanVariance }
                @test mean(marginal) === -2.0
                @test var(marginal) === 3.0
            end)
        end

        begin 
            r = randomvar(:r)
            setmarginal!(r, NormalMeanPrecision(-2.0, 3.0))
            subscribe!(getmarginal(r, IncludeAll()), (marginal) -> begin
                @test typeof(marginal) <: Marginal{ <: NormalMeanPrecision }
                @test mean(marginal) === -2.0
                @test precision(marginal) === 3.0
            end)
        end

        begin 
            rs = randomvar(:r, 2)
            setmarginals!(rs, NormalMeanVariance(-2.0, 3.0))
            subscribe!(getmarginals(rs, IncludeAll()), (marginals) -> begin
                foreach(marginals) do marginal
                    @test typeof(marginal) <: Marginal{ <: NormalMeanVariance }
                    @test mean(marginal) === -2.0
                    @test var(marginal) === 3.0
                end
            end)
        end

        begin 
            rs = randomvar(:r, 2)
            setmarginals!(rs, NormalMeanPrecision(-2.0, 3.0))
            subscribe!(getmarginals(rs, IncludeAll()), (marginals) -> begin
                foreach(marginals) do marginal
                    @test typeof(marginal) <: Marginal{ <: NormalMeanPrecision }
                    @test mean(marginal) === -2.0
                    @test precision(marginal) === 3.0
                end
            end)
        end

        # getmarginals is broken for matrices
        # begin 
        #     rs = randomvar(:r, 2, 2)
        #     setmarginals!(rs, NormalMeanVariance(-2.0, 3.0))
        #     subscribe!(getmarginals(rs, IncludeAll()), (marginals) -> begin
        #         foreach(marginals) do marginal
        #             @test typeof(marginal) <: Marginal{ <: NormalMeanPrecision }
        #             @test mean(marginal) === -2.0
        #             @test variance(marginal) === 3.0
        #         end
        #     end)
        # end

        # begin 
        #     rs = randomvar(:r, 2, 2)
        #     setmarginals!(rs, NormalMeanPrecision(-2.0, 3.0))
        #     subscribe!(getmarginals(rs, IncludeAll()), (marginals) -> begin
        #         foreach(marginals) do marginal
        #             @test typeof(marginal) <: Marginal{ <: NormalMeanPrecision }
        #             @test mean(marginal) === -2.0
        #             @test precision(marginal) === 3.0
        #         end
        #     end)
        # end


    end

end

end