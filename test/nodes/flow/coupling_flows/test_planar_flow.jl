module FlowNodeCouplingFlowsPlanarFlowTest

using Test
using ReactiveMP
using ReactiveMP: getdim, getu, getb, getw, getall, setu!, setb!, setw!
using ReactiveMP: forward, forward!, jacobian, jacobian!, inv_jacobian, inv_jacobian!
using ReactiveMP: det_jacobian, absdet_jacobian, logdet_jacobian, logdet_jacobian, logabsdet_jacobian

@testset "Planar Flow" begin
    @testset "Constructors" begin

        # check for unspecified input
        f = PlanarFlow()
        @test typeof(f) == ReactiveMP.PlanarFlowPlaceholder

        # check when creating placeholder
        f = ReactiveMP.PlanarFlowPlaceholder()
        @test typeof(f) == ReactiveMP.PlanarFlowPlaceholder

        # check when creating empty planar flow
        f = ReactiveMP.PlanarFlowEmpty(3)
        @test typeof(f) == ReactiveMP.PlanarFlowEmpty{3}
        @test getdim(f) == 3

        # check for specified dimensionality
        f = PlanarFlow(1)
        @test typeof(f.u) == Float64
        @test typeof(f.w) == Float64
        @test typeof(f.b) == Float64

        f = PlanarFlow(5)
        @test typeof(f.u) == Array{Float64, 1}
        @test typeof(f.w) == Array{Float64, 1}
        @test typeof(f.b) == Float64
        @test length(f.u) == 5
        @test length(f.w) == 5

        # check for prespecified parameters (univariate)
        f = PlanarFlow(1.0, 2.0, 3.0)
        @test typeof(f.u) == Float64
        @test typeof(f.w) == Float64
        @test typeof(f.b) == Float64
        @test f.u == 1.0
        @test f.w == 2.0
        @test f.b == 3.0

        # check for prespecified parameters (multivariates)
        f = PlanarFlow([2.0, 3.0], [5.0, 6.0], 3.0)
        @test typeof(f.u) == Array{Float64, 1}
        @test typeof(f.w) == Array{Float64, 1}
        @test typeof(f.b) == Float64
        @test f.u == [2.0, 3.0]
        @test f.w == [5.0, 6.0]
        @test f.b == 3.0

        # wrong constructors
        @test_throws MethodError PlanarFlow([2.0, 3.0], 2.0, 1.0)
        @test_throws MethodError PlanarFlow([5.0, 6.0], 1.0, 0.5)
        @test_throws MethodError PlanarFlow(2.0, [2.0, 3.0], 1.0)
        @test_throws MethodError PlanarFlow(1.0, [5.0, 6.0], 0.5)
        @test_throws AssertionError PlanarFlow([2.0, 3.0, 4.0], [2.0, 3.0], 1.0)
        @test_throws AssertionError PlanarFlow([1.0, 6.0, 8.0], [5.0, 9.0], -2.0)
        @test_throws AssertionError PlanarFlow([2.0, 3.0], [2.0, 3.0, 4.0], 1.0)
        @test_throws AssertionError PlanarFlow([5.0, 9.0], [1.0, 6.0, 8.0], -2.0)
    end

    @testset "Prepare-Compile" begin
        f = ReactiveMP.PlanarFlowPlaceholder()
        @test ReactiveMP.prepare(2, f) == ReactiveMP.PlanarFlowEmpty(2)
        @test ReactiveMP.prepare(5, f) == ReactiveMP.PlanarFlowEmpty(5)
        @test ReactiveMP.prepare(9, f) == ReactiveMP.PlanarFlowEmpty(9)

        f = ReactiveMP.PlanarFlowEmpty(1)
        params = [1.0, 2.0, 3.0]
        fc = ReactiveMP.compile(f)
        fcp = ReactiveMP.compile(f, params)
        @test typeof(fc) <: PlanarFlow
        @test typeof(fc.u) == Float64
        @test typeof(fc.w) == Float64
        @test typeof(fc.b) == Float64
        @test typeof(fcp) <: PlanarFlow
        @test typeof(fcp.u) == Float64
        @test typeof(fcp.w) == Float64
        @test typeof(fcp.b) == Float64
        @test fcp.u == 1.0
        @test fcp.w == 2.0
        @test fcp.b == 3.0

        f = ReactiveMP.PlanarFlowEmpty(2)
        params = [1.0, 2.0, 3.0, 4.0, 5.0]
        fc = ReactiveMP.compile(f)
        fcp = ReactiveMP.compile(f, params)
        @test typeof(fc) <: PlanarFlow
        @test typeof(fc.u) == Array{Float64, 1}
        @test typeof(fc.w) == Array{Float64, 1}
        @test typeof(fc.b) == Float64
        @test typeof(fcp) <: PlanarFlow
        @test typeof(fcp.u) == Array{Float64, 1}
        @test typeof(fcp.w) == Array{Float64, 1}
        @test typeof(fcp.b) == Float64
        @test fcp.u == [1.0, 2.0]
        @test fcp.w == [3.0, 4.0]
        @test fcp.b == 5.0
    end

    @testset "nr_params" begin
        for k in 1:10
            f = ReactiveMP.PlanarFlowEmpty(k)
            fc = compile(f)
            @test nr_params(f) == 2 * k + 1
            @test nr_params(fc) == 2 * k + 1
        end
    end

    @testset "Get-Set" begin

        # check get functions for univariate PlanarFlow
        f = PlanarFlow(1.0, 2.0, 3.0)
        @test getu(f) == f.u
        @test getw(f) == f.w
        @test getb(f) == f.b
        @test getall(f) == (f.u, f.w, f.b)
        @test getu(f) == 1.0
        @test getw(f) == 2.0
        @test getb(f) == 3.0
        @test getall(f) == (1.0, 2.0, 3.0)

        # check get functions for multivariate PlanarFlow
        f = PlanarFlow([1.0, 2.0], [3.0, 4.0], 5.0)
        @test getu(f) == f.u
        @test getw(f) == f.w
        @test getb(f) == f.b
        @test getall(f) == (f.u, f.w, f.b)
        @test getu(f) == [1.0, 2.0]
        @test getw(f) == [3.0, 4.0]
        @test getb(f) == 5.0
        @test getall(f) == ([1.0, 2.0], [3.0, 4.0], 5.0)

        # check setu! function (univariate)
        f = PlanarFlow(1.0, 2.0, 3.0)
        setu!(f, 4.0)
        @test getu(f) == 4.0
        @test getw(f) == 2.0
        @test getb(f) == 3.0

        # check setu! function (multivariate)
        f = PlanarFlow([1.0, 2.0], [3.0, 4.0], 5.0)
        setu!(f, [10.0, 11.0])
        @test getu(f) == [10.0, 11.0]
        @test getw(f) == [3.0, 4.0]
        @test getb(f) == 5.0

        # check setw! function (univariate)
        f = PlanarFlow(1.0, 2.0, 3.0)
        setw!(f, 4.0)
        @test getu(f) == 1.0
        @test getw(f) == 4.0
        @test getb(f) == 3.0

        # check setw! function (multivariate)
        f = PlanarFlow([1.0, 2.0], [3.0, 4.0], 5.0)
        setw!(f, [10.0, 11.0])
        @test getu(f) == [1.0, 2.0]
        @test getw(f) == [10.0, 11.0]
        @test getb(f) == 5.0

        # check setb! function (univariate)
        f = PlanarFlow(1.0, 2.0, 3.0)
        setb!(f, 4.0)
        @test getu(f) == 1.0
        @test getw(f) == 2.0
        @test getb(f) == 4.0

        # check setb! function (multivariate)
        f = PlanarFlow([1.0, 2.0], [3.0, 4.0], 5.0)
        setb!(f, 6.0)
        @test getu(f) == [1.0, 2.0]
        @test getw(f) == [3.0, 4.0]
        @test getb(f) == 6.0

        # check getdim
        f = ReactiveMP.PlanarFlowEmpty(2)
        @test ReactiveMP.getdim(f) == 2

        # check errors (univariate)
        f = PlanarFlow(1.0, 2.0, 3.0)
        @test_throws MethodError setu!(f, [1.0, 2.0])
        @test_throws MethodError setw!(f, [1.0, 2.0])
        @test_throws MethodError setb!(f, [1.0, 2.0])

        # check errors (multivariate)
        f = PlanarFlow([1.0, 2.0], [3.0, 4.0], 5.0)
        @test_throws MethodError setu!(f, 3.0)
        @test_throws MethodError setw!(f, 4.0)
        @test_throws MethodError setb!(f, [1.0, 2.0])
        @test_throws AssertionError setu!(f, [1.0, 3.0, 6.0])
        @test_throws AssertionError setw!(f, [1.0, 3.0, 6.0])
    end

    @testset "Base" begin

        # check base functions (univariate)
        f = PlanarFlow(1.0, 2.0, 3.0)
        @test eltype(f) == Float64
        @test eltype(PlanarFlow{Float64, Float64}) == Float64
        @test size(f) == 1
        @test length(f) == 1

        # check base functions (multivariate)
        f = PlanarFlow([1.0, 2.0], [3.0, 4.0], 5.0)
        @test eltype(f) == Float64
        @test eltype(PlanarFlow{Array{Float64, 1}, Float64}) == Float64
        @test size(f) == 2
        @test length(f) == 2

        # check base functions empty object
        f = ReactiveMP.PlanarFlowEmpty(2)
        @test size(f) == 2
        @test length(f) == 2
    end

    @testset "Forward-Backward" begin

        # check forward function (univariate)
        f = PlanarFlow(1.0, 2.0, -3.0)
        @test forward(f, 1.5) == 1.5
        @test forward(f, 2.5) == 3.464027580075817
        @test forward.(f, [1.5, 2.5]) == [1.5, 3.464027580075817]

        # check forward function (univariate function, multivariate input)
        f = PlanarFlow(1.0, 2.0, -3.0)
        @test forward(f, [1.5]) == 1.5
        @test forward(f, [2.5]) == 3.464027580075817
        @test forward.(f, [[1.5], [2.5]]) == [1.5, 3.464027580075817]

        # check forward function (multivariate)
        f = PlanarFlow([1.0, 2.0], [3.0, 4.0], 0.0)
        @test forward(f, [-4.0, 3.0]) == [-4.0, 3.0]
        @test forward(f, [-2.0, 1.5]) == [-2.0, 1.5]
        @test forward.(f, [[-4.0, 3.0], [-2.0, 1.5]]) == [[-4.0, 3.0], [-2.0, 1.5]]

        # check forward! function (multivariate)
        f = PlanarFlow([1.0, 2.0], [3.0, 4.0], 0.0)
        output = zeros(2)
        forward!(output, f, [-4.0, 3.0])
        @test output == [-4.0, 3.0]
        forward!(output, f, [-2.0, 1.5])
        @test output == [-2.0, 1.5]
    end

    @testset "Jacobian" begin

        # check jacobian function (univariate)
        f = PlanarFlow(1.0, 2.0, -3.0)
        @test jacobian(f, 1.5) == 3.0
        @test jacobian(f, 2.5) == 1.1413016497063289
        @test jacobian.(f, [1.5, 2.5]) == [3.0, 1.1413016497063289]

        # check jacobian function (univariate function, multivariate input)
        f = PlanarFlow(1.0, 2.0, -3.0)
        @test jacobian(f, [1.5]) == 3.0
        @test jacobian(f, [2.5]) == 1.1413016497063289
        @test jacobian.(f, [[1.5], [2.5]]) == [3.0, 1.1413016497063289]

        # check jacobian function (multivariate)
        f = PlanarFlow([1.0, 2.0], [3.0, 4.0], 0.0)
        @test jacobian(f, [-4.0, 3.0]) == [4.0 4.0; 6.0 9.0]
        @test jacobian(f, [-2.0, 1.5]) == [4.0 4.0; 6.0 9.0]
        @test jacobian.(f, [[-4.0, 3.0], [-2.0, 1.5]]) == [[4.0 4.0; 6.0 9.0], [4.0 4.0; 6.0 9.0]]

        # check jacobian! function (multivariate)
        f = PlanarFlow([1.0, 2.0], [3.0, 4.0], 0.0)
        output = zeros(2, 2)
        jacobian!(output, f, [-4.0, 3.0])
        @test output == [4.0 4.0; 6.0 9.0]
        jacobian!(output, f, [-2.0, 1.5])
        @test output == [4.0 4.0; 6.0 9.0]
    end

    @testset "Utility Jacobian" begin

        # check utility functions jacobian (univariate)
        f = PlanarFlow(1.0, 2.0, -3.0)
        @test det_jacobian(f, 1.5) == 3.0
        @test inv_jacobian(f, 1.5) == 1 / 3
        @test absdet_jacobian(f, 1.5) == 3.0
        @test logabsdet_jacobian(f, 1.5) == log(3.0)

        # check utility functions jacobian (multivariate)
        f = PlanarFlow([1.0, 2.0], [3.0, 4.0], 0.0)
        @test det_jacobian(f, [-4.0, 3.0]) == 12.0
        @test inv_jacobian(f, [-4.0, 3.0]) ≈ [0.75 -1/3; -0.5 1/3]
        @test absdet_jacobian(f, [-4.0, 3.0]) == 12.0
        @test logabsdet_jacobian(f, [-4.0, 3.0]) == (log(12.0), 1)
    end
end

end
