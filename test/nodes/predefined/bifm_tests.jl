@testitem "BIFM Node" begin
    using ReactiveMP, Distributions, LinearAlgebra, Random, Test
    import ReactiveMP:
        getA, getB, getC, getH, getμu, getΣu, default_meta, setH!, setBHBt!, setξz!, setΛz!, setξztilde!, setΛztilde!, setμu!, setΣu!, getBHBt, getξz, getΛz, getξztilde, getΛztilde

    @testset "BIFMMeta constructors" begin
        A = [1.0 0.1; 0.0 1.0]
        B = [0.5 0.0; 0.0 0.5]
        C = [1.0 0.0; 0.0 1.0]

        meta1 = BIFMMeta(A, B, C)
        @test meta1 isa BIFMMeta{Float64}
        @test getA(meta1) === A
        @test getB(meta1) === B
        @test getC(meta1) === C
        @test getH(meta1) === nothing
        @test getμu(meta1) === nothing

        # test dimension checks
        @test_throws AssertionError BIFMMeta(ones(2, 2), ones(3, 2), ones(2, 2))

        # constructor with input priors
        μu = randn(2)
        Σu = diageye(2)
        meta2 = BIFMMeta(A, B, C, μu, Σu)
        @test getμu(meta2) == μu
        @test getΣu(meta2) == Σu
        @test meta2 isa BIFMMeta{Float64}
    end

    @testset "BIFMMeta setters and getters" begin
        A = [1.0 0.0; 0.0 1.0]
        B = [0.2 0.3; 0.1 0.4]
        C = [1.0 0.0; 0.0 1.0]
        meta = BIFMMeta(A, B, C)

        H = rand(2, 2)
        BHBt = rand(2, 2)
        ξz = rand(2)
        Λz = diageye(2)
        ξztilde = rand(2)
        Λztilde = diageye(2)
        μu = rand(2)
        Σu = diageye(2)

        setH!(meta, H)
        setBHBt!(meta, BHBt)
        setξz!(meta, ξz)
        setΛz!(meta, Λz)
        setξztilde!(meta, ξztilde)
        setΛztilde!(meta, Λztilde)
        setμu!(meta, μu)
        setΣu!(meta, Σu)

        @test getH(meta) === H
        @test getBHBt(meta) === BHBt
        @test getξz(meta) === ξz
        @test getΛz(meta) === Λz
        @test getξztilde(meta) === ξztilde
        @test getΛztilde(meta) === Λztilde
        @test getμu(meta) === μu
        @test getΣu(meta) === Σu
    end

    @testset "BIFM node construction and interface structure" begin
        @test sdtype(BIFM) == Deterministic()

        # TODO: test for interface indices and names. Problem: Creating factornode with node = factornode(BIFM, interfaces, factorizations) and then call getinterfaces and interfaceindex, but there is no method matching collect_factorisation(::Type{ReactiveMP.BIFM}, ::Vector{Vector{Symbol}})
    end
end
