module FlowNodeCouplingFlowsRadialFlowTest

using Test
using ReactiveMP 

@testset "Radial Flow" begin

    @testset "Constructor" begin
        
        # check for unspecified input
        f = RadialFlow()
        @test typeof(f) == ReactiveMP.RadialFlowPlaceholder

        # check when creating placeholder
        f = ReactiveMP.RadialFlowPlaceholder()
        @test typeof(f) == ReactiveMP.RadialFlowPlaceholder

        # check when creating empty planar flow
        f = ReactiveMP.RadialFlowEmpty(3)
        @test typeof(f) == ReactiveMP.RadialFlowEmpty{3}
        @test getdim(f) == 3

        # check for specified dimensionality
        f = RadialFlow(3)
        @test typeof(f.z0) == Array{Float64,1}
        @test typeof(f.α)  == Float64
        @test typeof(f.β)  == Float64
        @test length(f.z0) == 3

        f = RadialFlow(5)
        @test typeof(f.z0) == Array{Float64,1}
        @test typeof(f.α)  == Float64
        @test typeof(f.β)  == Float64
        @test length(f.z0) == 5

        # check for prespecified parameters (univariate)
        f = RadialFlow(1.0, 2.0, 3.0)
        @test typeof(f.z0) == Float64
        @test typeof(f.α)  == Float64
        @test typeof(f.β)  == Float64
        @test f.z0 == 1.0
        @test f.α  == 2.0
        @test f.β  == 3.0

        # check for prespecified parameters (multivariates)
        f = RadialFlow([2.0, 3.0], 5.0, 3.0)
        @test typeof(f.z0) == Array{Float64,1}
        @test typeof(f.α)  == Float64
        @test typeof(f.β)  == Float64
        @test f.z0 == [2.0, 3.0]
        @test f.α  == 5.0
        @test f.β  == 3.0

        # wrong constructors
        @test_throws MethodError RadialFlow(1.0, 2.0, [2.0, 3.0])
        @test_throws MethodError RadialFlow(0.5, 1.0, [5.0, 6.0])
        @test_throws MethodError RadialFlow(2.0, [2.0, 3.0], 1.0)
        @test_throws MethodError RadialFlow(1.0, [5.0, 6.0], 0.5)
        @test_throws AssertionError RadialFlow(1.0, -2.0, 1.0)
        @test_throws AssertionError RadialFlow(2.0, -1.0, -2.0)
        @test_throws AssertionError RadialFlow([2.0, 3.0], -1.0, 1.0)
        @test_throws AssertionError RadialFlow([5.0, 9.0], -5.0, -2.0)
        
    end

    @testset "Prepare-Compile" begin

        f = ReactiveMP.RadialFlowPlaceholder()
        @test ReactiveMP.prepare(2, f) == ReactiveMP.RadialFlowEmpty(2)
        @test ReactiveMP.prepare(5, f) == ReactiveMP.RadialFlowEmpty(5)
        @test ReactiveMP.prepare(9, f) == ReactiveMP.RadialFlowEmpty(9)

        f = ReactiveMP.RadialFlowEmpty(1)
        params = [1.0, 2.0, 3.0]
        fc = ReactiveMP.compile(f)
        fcp = ReactiveMP.compile(f, params)
        @test typeof(fc)     <: RadialFlow
        @test typeof(fc.z0)  == Float64
        @test typeof(fc.α)   == Float64
        @test typeof(fc.β)   == Float64
        @test typeof(fcp)    <: RadialFlow
        @test typeof(fcp.z0) == Float64
        @test typeof(fcp.α)  == Float64
        @test typeof(fcp.β)  == Float64
        @test fcp.z0         == 1.0
        @test fcp.α          == 2.0
        @test fcp.β          == 3.0

        f = ReactiveMP.RadialFlowEmpty(2)
        params = [1.0, 2.0, 3.0, 4.0]
        fc = ReactiveMP.compile(f)
        fcp = ReactiveMP.compile(f, params)
        @test typeof(fc)     <: RadialFlow
        @test typeof(fc.z0)  == Array{Float64,1}
        @test typeof(fc.α)   == Float64
        @test typeof(fc.β)   == Float64
        @test typeof(fcp)    <: RadialFlow
        @test typeof(fcp.z0) == Array{Float64,1}
        @test typeof(fcp.α)  == Float64
        @test typeof(fcp.β)  == Float64
        @test fcp.z0         == [1.0, 2.0]
        @test fcp.α          == 3.0
        @test fcp.β          == 4.0

    end

    @testset "nr_params" begin
        
        for k = 1:10
            f = ReactiveMP.RadialFlowEmpty(k)
            fc = compile(f)
            @test nr_params(f)  == k + 2
            @test nr_params(fc) == k + 2
        end

    end

    @testset "Get-Set" begin
        
        # check get functions for univariate RadialFlow
        f = RadialFlow(1.0, 2.0, 3.0)
        @test getz0(f)  == f.z0
        @test getα(f)   == f.α
        @test getβ(f)   == f.β
        @test getall(f) == (f.z0, f.α, f.β)
        @test getz0(f)  == 1.0
        @test getα(f)   == 2.0
        @test getβ(f)   == 3.0
        @test getall(f) == (1.0, 2.0, 3.0)

        # check get functions for multivariate RadialFlow
        f = RadialFlow([1.0, 2.0], 3.0, 4.0)
        @test getz0(f)  == f.z0
        @test getα(f)   == f.α
        @test getβ(f)   == f.β
        @test getall(f) == (f.z0, f.α, f.β)
        @test getz0(f)  == [1.0, 2.0]
        @test getα(f)   == 3.0
        @test getβ(f)   == 4.0
        @test getall(f) == ([1.0, 2.0], 3.0, 4.0)

        # check setz0! function (univariate)
        f = RadialFlow(1.0, 2.0, 3.0)
        setz0!(f, 4.0)
        @test getz0(f) == 4.0
        @test getα(f)  == 2.0
        @test getβ(f)  == 3.0

        # check setz0! function (multivariate)
        f = RadialFlow([1.0, 2.0], 3.0, 4.0)
        setz0!(f, [10.0, 11.0])
        @test getz0(f) == [10.0, 11.0]
        @test getα(f)  == 3.0
        @test getβ(f)  == 4.0

        # check setα! function (univariate)
        f = RadialFlow(1.0, 2.0, 3.0)
        setα!(f, 4.0)
        @test getz0(f) == 1.0
        @test getα(f)  == 4.0
        @test getβ(f)  == 3.0

        # check setα! function (multivariate)
        f = RadialFlow([1.0, 2.0], 3.0, 5.0)
        setα!(f, 4.0)
        @test getz0(f) == [1.0, 2.0]
        @test getα(f)  == 4.0
        @test getβ(f)  == 5.0

        # check setβ! function (univariate)
        f = RadialFlow(1.0, 2.0, 3.0)
        setβ!(f, 4.0)
        @test getz0(f) == 1.0
        @test getα(f)  == 2.0
        @test getβ(f)  == 4.0

        # check setβ! function (multivariate)
        f = RadialFlow([1.0, 2.0], 3.0, 5.0)
        setβ!(f, 6.0)
        @test getz0(f) == [1.0, 2.0]
        @test getα(f)  == 3.0
        @test getβ(f)  == 6.0

        # check getdim
        f = ReactiveMP.RadialFlowEmpty(2)
        @test ReactiveMP.getdim(f) == 2

        # check errors (univariate)
        f = RadialFlow(1.0, 2.0, 3.0)
        @test_throws MethodError setz0!(f, [1.0, 2.0])
        @test_throws MethodError setα!(f, [1.0, 2.0])
        @test_throws MethodError setβ!(f, [1.0, 2.0])

        # check errors (multivariate)
        f = RadialFlow([1.0, 2.0], 3.0, 5.0)
        @test_throws MethodError setz0!(f, 3.0)
        @test_throws MethodError setα!(f, [4.0, 3.0])
        @test_throws MethodError setβ!(f, [1.0, 2.0])
        @test_throws AssertionError setz0!(f, [1.0, 3.0, 6.0])

    end

    @testset "Base" begin
        
        # check base functions (univariate)
        f = RadialFlow(1.0, 2.0, 3.0)
        @test eltype(f) == Float64
        @test eltype(RadialFlow{Float64,Float64}) == Float64
        @test size(f) == 1
        @test length(f) == 1

        # check base functions (multivariate)
        f = RadialFlow([1.0, 2.0], 3.0, 5.0)
        @test eltype(f) == Float64
        @test eltype(RadialFlow{Array{Float64,1},Float64}) == Float64
        @test size(f) == 2
        @test length(f) == 2

        # check base functions empty object
        f = ReactiveMP.RadialFlowEmpty(2)
        @test size(f)   == 2
        @test length(f) == 2

    end

    @testset "Forward-Backward" begin
        
        # check forward function (univariate)
        f = RadialFlow(1.0, 2.0, -3.0)
        @test forward(f, 1.5) == 0.9
        @test forward(f, 2.5) == 1.2142857142857144
        @test forward.(f, [1.5, 2.5]) == [0.9, 1.2142857142857144]

        # check forward function (multivariate)
        f = RadialFlow([1.0, 2.0], 3.0, 1.0)
        @test forward(f, [-4.0, 3.0]) == [-4.617358680468466, 3.1234717360936934]
        @test forward(f, [-2.0, 1.5]) == [-2.4965751817893183, 1.4172374697017802]
        @test forward.(f, [[-4.0, 3.0], [-2.0, 1.5]]) == [[-4.617358680468466, 3.1234717360936934], [-2.4965751817893183, 1.4172374697017802]]

        # check forward! function (multivariate)
        f = RadialFlow([1.0, 2.0], 3.0, 1.0)
        output = zeros(2)
        forward!(output, f, [-4.0, 3.0]) 
        @test output == [-4.617358680468466, 3.1234717360936934]
        forward!(output, f, [-2.0, 1.5]) 
        @test output == [-2.4965751817893183, 1.4172374697017802]

    end

    @testset "Jacobian" begin
        
        # check jacobian function (univariate)
        f = RadialFlow(1.0, 2.0, -3.0)
        @test jacobian(f, 1.5) == 0.03999999999999987
        @test jacobian(f, 2.5) == 0.5102040816326531
        @test jacobian.(f, [1.5, 2.5]) == [0.03999999999999987, 0.5102040816326531]

        # check jacobian function (multivariate)
        f = RadialFlow([1.0, 2.0], 3.0, 1.0)
        @test jacobian(f, [-4.0, 3.0]) == [1.0487256521978074 0.014949216779177184; 0.014949216779177184 1.120481892737858]
        @test jacobian(f, [-2.0, 1.5]) == [1.084447783638529 -0.013512879492985073; -0.013512879492985073 1.1632729140142752]
        @test jacobian.(f, [[-4.0, 3.0], [-2.0, 1.5]]) == [[1.0487256521978074 0.014949216779177184; 0.014949216779177184 1.120481892737858], [1.084447783638529 -0.013512879492985073; -0.013512879492985073 1.1632729140142752]]

        # check jacobian! function (multivariate)
        f = RadialFlow([1.0, 2.0], 3.0, 1.0)
        output = zeros(2,2)
        jacobian!(output, f, [-4.0, 3.0]) 
        @test output == [1.0487256521978074 0.014949216779177184; 0.014949216779177184 1.120481892737858]
        jacobian!(output, f, [-2.0, 1.5]) 
        @test output == [1.084447783638529 -0.013512879492985073; -0.013512879492985073 1.1632729140142752]

    end

    @testset "Utility Jacobian" begin
        
        # check utility functions jacobian (univariate)
        f = RadialFlow(1.0, 2.0, -3.0)
        @test det_jacobian(f, 1.5) == 0.03999999999999987
        @test inv_jacobian(f, 1.5) ≈ 25
        @test absdet_jacobian(f, 1.5) == 0.03999999999999987
        @test logabsdet_jacobian(f, 1.5) == -3.218875824868204

        # check utility functions jacobian (multivariate)
        f = RadialFlow([1.0, 2.0], 3.0, 1.0)
        @test det_jacobian(f, [-4.0, 3.0]) == 1.1748546246550329
        @test inv_jacobian(f, [-4.0, 3.0]) ≈ [0.9537196085574075 -0.012724311983337218; -0.012724311983337218 0.8926429110373889]
        @test absdet_jacobian(f, [-4.0, 3.0]) == 1.1748546246550329
        @test logabsdet_jacobian(f, [-4.0, 3.0]) == (0.16114441624386985, 1.0)

    end

end

end