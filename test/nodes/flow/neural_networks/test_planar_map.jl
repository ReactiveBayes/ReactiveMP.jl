module FlowNodeNeuralNetworksPlanarMapTest

using Test
using ReactiveMP 

@testset "Planar Map" begin

    @testset "Constructor" begin
        
        # check for unspecified univariate input
        f = PlanarMap()
        @test typeof(f.u) == Float64
        @test typeof(f.w) == Float64
        @test typeof(f.b) == Float64

        # check for specified dimensionality
        f = PlanarMap(3)
        @test typeof(f.u) == Array{Float64,1}
        @test typeof(f.w) == Array{Float64,1}
        @test typeof(f.b) == Float64
        @test length(f.u) == 3
        @test length(f.w) == 3

        f = PlanarMap(5)
        @test typeof(f.u) == Array{Float64,1}
        @test typeof(f.w) == Array{Float64,1}
        @test typeof(f.b) == Float64
        @test length(f.u) == 5
        @test length(f.w) == 5

        # check for prespecified parameters (univariate)
        f = PlanarMap(1.0, 2.0, 3.0)
        @test typeof(f.u) == Float64
        @test typeof(f.w) == Float64
        @test typeof(f.b) == Float64
        @test f.u == 1.0
        @test f.w == 2.0
        @test f.b == 3.0

        # check for prespecified parameters (multivariates)
        f = PlanarMap([2.0, 3.0], [5.0, 6.0], 3.0)
        @test typeof(f.u) == Array{Float64,1}
        @test typeof(f.w) == Array{Float64,1}
        @test typeof(f.b) == Float64
        @test f.u == [2.0, 3.0]
        @test f.w == [5.0, 6.0]
        @test f.b == 3.0

        # wrong constructors
        @test_throws MethodError PlanarMap([2.0, 3.0], 2.0, 1.0)
        @test_throws MethodError PlanarMap([5.0, 6.0], 1.0, 0.5)
        @test_throws MethodError PlanarMap(2.0, [2.0, 3.0], 1.0)
        @test_throws MethodError PlanarMap(1.0, [5.0, 6.0], 0.5)
        @test_throws AssertionError PlanarMap([2.0, 3.0, 4.0], [2.0, 3.0], 1.0)
        @test_throws AssertionError PlanarMap([1.0, 6.0, 8.0], [5.0, 9.0], -2.0)
        @test_throws AssertionError PlanarMap([2.0, 3.0], [2.0, 3.0, 4.0], 1.0)
        @test_throws AssertionError PlanarMap([5.0, 9.0], [1.0, 6.0, 8.0], -2.0)
        
    end

    @testset "Get-Set" begin
        
        # check get functions for univariate PlanarMap
        f = PlanarMap(1.0, 2.0, 3.0)
        @test getu(f) == f.u
        @test getw(f) == f.w
        @test getb(f) == f.b
        @test getall(f) == (f.u, f.w, f.b)
        @test getu(f) == 1.0
        @test getw(f) == 2.0
        @test getb(f) == 3.0
        @test getall(f) == (1.0, 2.0, 3.0)

        # check get functions for multivariate PlanarMap
        f = PlanarMap([1.0, 2.0], [3.0, 4.0], 5.0)
        @test getu(f) == f.u
        @test getw(f) == f.w
        @test getb(f) == f.b
        @test getall(f) == (f.u, f.w, f.b)
        @test getu(f) == [1.0, 2.0]
        @test getw(f) == [3.0, 4.0]
        @test getb(f) == 5.0
        @test getall(f) == ([1.0, 2.0], [3.0, 4.0], 5.0)

        # check setu! function (univariate)
        f = PlanarMap(1.0, 2.0, 3.0)
        setu!(f, 4.0)
        @test getu(f) == 4.0
        @test getw(f) == 2.0
        @test getb(f) == 3.0

        # check setu! function (multivariate)
        f = PlanarMap([1.0, 2.0], [3.0, 4.0], 5.0)
        setu!(f, [10.0, 11.0])
        @test getu(f) == [10.0, 11.0]
        @test getw(f) == [3.0, 4.0]
        @test getb(f) == 5.0

        # check setw! function (univariate)
        f = PlanarMap(1.0, 2.0, 3.0)
        setw!(f, 4.0)
        @test getu(f) == 1.0
        @test getw(f) == 4.0
        @test getb(f) == 3.0

        # check setw! function (multivariate)
        f = PlanarMap([1.0, 2.0], [3.0, 4.0], 5.0)
        setw!(f, [10.0, 11.0])
        @test getu(f) == [1.0, 2.0]
        @test getw(f) == [10.0, 11.0]
        @test getb(f) == 5.0

        # check setb! function (univariate)
        f = PlanarMap(1.0, 2.0, 3.0)
        setb!(f, 4.0)
        @test getu(f) == 1.0
        @test getw(f) == 2.0
        @test getb(f) == 4.0

        # check setb! function (multivariate)
        f = PlanarMap([1.0, 2.0], [3.0, 4.0], 5.0)
        setb!(f, 6.0)
        @test getu(f) == [1.0, 2.0]
        @test getw(f) == [3.0, 4.0]
        @test getb(f) == 6.0

        # check errors (univariate)
        f = PlanarMap(1.0, 2.0, 3.0)
        @test_throws MethodError setu!(f, [1.0, 2.0])
        @test_throws MethodError setw!(f, [1.0, 2.0])
        @test_throws MethodError setb!(f, [1.0, 2.0])

        # check errors (multivariate)
        f = PlanarMap([1.0, 2.0], [3.0, 4.0], 5.0)
        @test_throws MethodError setu!(f, 3.0)
        @test_throws MethodError setw!(f, 4.0)
        @test_throws MethodError setb!(f, [1.0, 2.0])
        @test_throws AssertionError setu!(f, [1.0, 3.0, 6.0])
        @test_throws AssertionError setw!(f, [1.0, 3.0, 6.0])

    end

    @testset "Base" begin
        
        # check base functions (univariate)
        f = PlanarMap()
        @test eltype(f) == Float64
        @test eltype(PlanarMap{Float64,Float64}) == Float64
        @test size(f) == 1
        @test length(f) == 1

        # check base functions (multivariate)
        f = PlanarMap([1.0, 2.0], [3.0, 4.0], 5.0)
        @test eltype(f) == Float64
        @test eltype(PlanarMap{Array{Float64,1},Float64}) == Float64
        @test size(f) == 2
        @test length(f) == 2

    end

    @testset "Forward-Backward" begin
        
        # check forward function (univariate)
        f = PlanarMap(1.0, 2.0, -3.0)
        @test forward(f, 1.5) == 1.5
        @test forward(f, 2.5) == 3.464027580075817
        @test forward.(f, [1.5, 2.5]) == [1.5, 3.464027580075817]

        # check forward function (multivariate)
        f = PlanarMap([1.0, 2.0], [3.0, 4.0], 0.0)
        @test forward(f, [-4.0, 3.0]) == [-4.0, 3.0]
        @test forward(f, [-2.0, 1.5]) == [-2.0, 1.5]
        @test forward.(f, [[-4.0, 3.0], [-2.0, 1.5]]) == [[-4.0, 3.0], [-2.0, 1.5]]

        # check forward! function (multivariate)
        f = PlanarMap([1.0, 2.0], [3.0, 4.0], 0.0)
        output = zeros(2)
        forward!(output, f, [-4.0, 3.0]) 
        @test output == [-4.0, 3.0]
        forward!(output, f, [-2.0, 1.5]) 
        @test output == [-2.0, 1.5]

    end

    @testset "Jacobian" begin
        
        # check forward function (univariate)
        f = PlanarMap(1.0, 2.0, -3.0)
        @test jacobian(f, 1.5) == 3.0
        @test jacobian(f, 2.5) == 1.1413016497063289
        @test jacobian.(f, [1.5, 2.5]) == [3.0, 1.1413016497063289]

        # check forward function (multivariate)
        f = PlanarMap([1.0, 2.0], [3.0, 4.0], 0.0)
        @test jacobian(f, [-4.0, 3.0]) == [4.0 4.0; 6.0 9.0]
        @test jacobian(f, [-2.0, 1.5]) == [4.0 4.0; 6.0 9.0]
        @test jacobian.(f, [[-4.0, 3.0], [-2.0, 1.5]]) == [[4.0 4.0; 6.0 9.0], [4.0 4.0; 6.0 9.0]]

        # check forward! function (multivariate)
        f = PlanarMap([1.0, 2.0], [3.0, 4.0], 0.0)
        output = zeros(2,2)
        jacobian!(output, f, [-4.0, 3.0]) 
        @test output == [4.0 4.0; 6.0 9.0]
        jacobian!(output, f, [-2.0, 1.5]) 
        @test output == [4.0 4.0; 6.0 9.0]

    end

    @testset "Utility Jacobian" begin
        
        # check utility functions jacobian (univariate)
        f = PlanarMap(1.0, 2.0, -3.0)
        @test det_jacobian(f, 1.5) == 3.0
        @test inv_jacobian(f, 1.5) == 1/3
        @test absdet_jacobian(f, 1.5) == 3.0
        @test logabsdet_jacobian(f, 1.5) == log(3.0)

        # check utility functions jacobian (multivariate)
        f = PlanarMap([1.0, 2.0], [3.0, 4.0], 0.0)
        @test det_jacobian(f, [-4.0, 3.0]) == 12.0
        @test inv_jacobian(f, [-4.0, 3.0]) ≈ [0.75 -1/3; -0.5 1/3]
        @test absdet_jacobian(f, [-4.0, 3.0]) == 12.0
        @test logabsdet_jacobian(f, [-4.0, 3.0]) == (log(12.0), 1)

    end

end

end