
module ReactiveMPAddonsTest

using Test
using ReactiveMP
using Distributions

using ReactiveMP: multiply_addons

@testset "Addons" begin
    @testset "addonlogscale" begin
        @testset "creation" begin
            addon1 = AddonLogScale()
            addon2 = AddonLogScale(2)
            addon3 = AddonLogScale(3.0)

            @test addon1.logscale == nothing
            @test addon2.logscale == 2
            @test addon3.logscale == 3.0
        end

        @testset "getlogscale" begin
            message  = Message(Normal(1, 0), false, false, (AddonLogScale(3),))
            marginal = Marginal(Normal(1, 0), false, false, (AddonLogScale(4.0),))

            @test getlogscale(message) == 3
            @test getlogscale(marginal) == 4.0
        end

        @testset "multiply_addons" begin
            left_addons = (AddonLogScale(5),)
            right_addons = (AddonLogScale(6.0),)
            new_dist = vague(Bernoulli)
            left_dist = vague(Bernoulli)
            right_dist = vague(Bernoulli)

            @test multiply_addons(left_addons, right_addons, new_dist, left_dist, right_dist) == (AddonLogScale(11.0 - log(2)),)
            @test multiply_addons(AddonLogScale(5), AddonLogScale(6.0), new_dist, left_dist, right_dist) == AddonLogScale(11.0 - log(2))
            @test multiply_addons(AddonLogScale(5), nothing, new_dist, left_dist, missing) == AddonLogScale(5)
            @test multiply_addons(nothing, AddonLogScale(6.0), new_dist, missing, right_dist) == AddonLogScale(6.0)
            @test multiply_addons(nothing, nothing, new_dist, left_dist, missing) == nothing
            @test multiply_addons(nothing, nothing, new_dist, missing, right_dist) == nothing
        end
    end
end

end
