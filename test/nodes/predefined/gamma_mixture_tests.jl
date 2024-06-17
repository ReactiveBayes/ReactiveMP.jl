@testitem "GammaShapeLikelihood insupport" begin
    using BayesBase
    import ReactiveMP: GammaShapeLikelihood
    
    for p in [1.0, 2.0, 3.0], γ in [1.0, 2.0, 3.0], s in [1.0, 2.0]
        @test insupport(GammaShapeLikelihood(p, γ), s)
        @test !insupport(GammaShapeLikelihood(p, γ), -s)
    end

end