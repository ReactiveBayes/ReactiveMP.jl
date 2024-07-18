@testitem "Memory addon" begin
    using ExponentialFamily, BayesBase

    import ReactiveMP: AddonMemory

    @testset "Creation" begin
        addon = AddonMemory()

        @test occursin("memory", string(addon))
    end

    @testset "Simple application and printing" begin
        import ReactiveMP: MessageMapping, message_mapping_addon

        mapping = MessageMapping(NormalMeanVariance, Val(:out), Marginalisation(), Val((:x, :y)), Val((:z, :k)), "meta", AddonMemory(), nothing, nothing)

        messages = (Gamma(1.0, 1.0), NormalMeanVariance(0.0, 1.0))
        marginals = (PointMass(1.0), NormalMeanPrecision(0.0, 1.0))
        result = MvNormalMeanCovariance(ones(2), diageye(2))

        addon = message_mapping_addon(AddonMemory(), mapping, messages, marginals, result)

        displayed = repr(addon)

        @test occursin(r"node: ExponentialFamily.NormalMeanVariance", displayed)
        @test occursin(r"interface: .*:out.*", displayed)
        @test occursin(r"local constraint:.*Marginalisation()", displayed)
        @test occursin(r"messages on.*(:x, :y).*edges", displayed)
        @test occursin(repr(messages), displayed)
        @test occursin(r"marginals on.*(:z, :k).*edges", displayed)
        @test occursin(repr(marginals), displayed)
        @test occursin(r"meta: meta", displayed)
        @test occursin("result: $(repr(result))", displayed)
    end
end
