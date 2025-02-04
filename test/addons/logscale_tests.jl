@testitem "Logscale addon" begin
    import ReactiveMP: AddonLogScale, AddonMemory, AddonDebug

    @testset "Error handling" begin
        @test_throws "Log-scale addon is not available. Make sure to include AddonLogScale in the addons. Currently, log scale factors are only supported for very specific nodes and messages in sum-product updates. Extensions to variational message passing are not yet supported." getlogscale(nothing)
    end

    @testset "Simple application" begin
        addon = AddonLogScale(0.0)
        @test getlogscale(addon) == 0.0
    end

end
