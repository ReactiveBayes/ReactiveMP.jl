@testitem "Debug addon" begin
    import ReactiveMP: AddonDebug

    @testset "Creation" begin
        addon = AddonDebug(x -> x == π)

        @test addon(π)
        @test addon(3.14) == false
    end

    @testset "Simple application and printing" begin
        import ReactiveMP: multiply_addons

        addon = AddonDebug(x -> any(params(x) .== 3.14))
        @test_throws ErrorException multiply_addons(addon, addon, NormalMeanVariance(0.0, 3.14), Missing(), Missing())
        @test multiply_addons(addon, addon, NormalMeanVariance(0.0, 3.00), Missing(), Missing()) == addon
    end
end
