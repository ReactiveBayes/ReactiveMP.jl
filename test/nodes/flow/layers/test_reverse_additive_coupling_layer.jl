module FlowNodeLayersReverseAdditiveCouplingLayerTest

using Test
using ReactiveMP 

@testset "Reverse Additive Coupling Layer" begin

    @testset "Constructor" begin
        
        # check for layer with univariate function
        f = PlanarFlow()
        layer = ReverseAdditiveCouplingLayer(f)
        @test layer.f == f
        @test layer.f.u == f.u
        @test layer.f.w == f.w
        @test layer.f.b == f.b

        # check for layer with multivariate function
        f = PlanarFlow(3)
        layer = ReverseAdditiveCouplingLayer(f)
        @test layer.f == f
        @test layer.f.u == f.u
        @test layer.f.w == f.w
        @test layer.f.b == f.b

    end

    @testset "Get" begin
        
        # check get functions for univariate PlanarFlow
        f = PlanarFlow()
        layer = ReverseAdditiveCouplingLayer(f)
        @test getf(layer) == layer.f
        @test getflow(layer) == layer.f
        @test getf(layer) == f
        @test getflow(layer) == f

        # check get functions for multivariate PlanarFlow
        f = PlanarFlow(3)
        layer = ReverseAdditiveCouplingLayer(f)
        @test getf(layer) == layer.f
        @test getflow(layer) == layer.f
        @test getf(layer) == f
        @test getflow(layer) == f

    end

    @testset "Base" begin
        
        # check base functions (univariate)
        f = PlanarFlow()
        layer = ReverseAdditiveCouplingLayer(f)
        @test eltype(layer) == Float64
        @test eltype(ReverseAdditiveCouplingLayer{PlanarFlow{Float64,Float64}}) == Float64

        # check base functions (multivariate)
        f = PlanarFlow(3)
        layer = ReverseAdditiveCouplingLayer(f)
        @test eltype(layer) == Float64
        @test eltype(ReverseAdditiveCouplingLayer{PlanarFlow{Array{Float64,1},Float64}}) == Float64

    end

    @testset "Forward-Backward" begin
        
        # check forward function
        f = PlanarFlow(1.0, 2.0, -3.0)
        layer = ReverseAdditiveCouplingLayer(f)
        @test forward(layer, [5.0, 1.5]) == [6.5, 1.5]
        @test forward(layer, [4.0, 2.5]) == [7.4640275800758165, 2.5]
        @test forward.(layer, [[5.0, 1.5], [4.0, 2.5]]) == [[6.5, 1.5], [7.4640275800758165, 2.5]]

        # check forward! function
        f = PlanarFlow(1.0, 2.0, -3.0)
        layer = ReverseAdditiveCouplingLayer(f)
        output = zeros(2)
        forward!(output, layer, [5.0, 1.5]) 
        @test output == [6.5, 1.5]
        forward!(output, layer, [4.0, 2.5]) 
        @test output == [7.4640275800758165, 2.5]

        # check backward function
        f = PlanarFlow(1.0, 2.0, -3.0)
        layer = ReverseAdditiveCouplingLayer(f)
        @test backward(layer, [6.5, 1.5]) == [5.0, 1.5]
        @test backward(layer, [7.4640275800758165, 2.5]) ≈ [4.0, 2.5]
        @test backward.(layer, [[6.5, 1.5], [7.4640275800758165, 2.5]]) ≈ [[5.0, 1.5], [4.0, 2.5]]

        # check backward! function
        f = PlanarFlow(1.0, 2.0, -3.0)
        layer = ReverseAdditiveCouplingLayer(f)
        output = zeros(2)
        backward!(output, layer, [6.5, 1.5]) 
        @test output == [5.0, 1.5]
        backward!(output, layer, [7.4640275800758165, 2.5]) 
        @test output ≈ [4.0, 2.5]

    end

    @testset "Jacobian" begin
        
        # check jacobian function
        f = PlanarFlow(1.0, 2.0, -3.0)
        layer = ReverseAdditiveCouplingLayer(f)
        @test jacobian(layer, [3.0, 1.5]) == [1.0 1.0197320743308804; 0.0 1.0]
        @test jacobian(layer, [2.5, 5.0]) == [1.0 1.1413016497063289; 0.0 1.0]
        @test jacobian.(layer, [[3.0, 1.5], [2.5, 5.0]]) == [[1.0 1.0197320743308804; 0.0 1.0], [1.0 1.1413016497063289; 0.0 1.0]]

        # check jacobian function
        f = PlanarFlow(1.0, 2.0, -3.0)
        layer = ReverseAdditiveCouplingLayer(f)
        @test inv_jacobian(layer, [3.0, 1.5]) == [1.0 -1.0197320743308804; 0.0 1.0]
        @test inv_jacobian(layer, [2.5, 5.0]) == [1.0 -1.1413016497063289; 0.0 1.0]
        @test inv_jacobian.(layer, [[3.0, 1.5], [2.5, 5.0]]) == [[1.0 -1.0197320743308804; 0.0 1.0], [1.0 -1.1413016497063289; 0.0 1.0]]        

    end

    @testset "Utility Jacobian" begin
        
        # check utility functions jacobian (univariate)
        f = PlanarFlow(1.0, 2.0, -3.0)
        layer = ReverseAdditiveCouplingLayer(f)
        @test det_jacobian(layer, [1.5, 6.9]) == 1.0
        @test det_jacobian(layer, [2.5, 6.4]) == 1.0
        @test absdet_jacobian(layer, [1.5, 6.9]) == 1.0
        @test absdet_jacobian(layer, [2.5, 6.4]) == 1.0
        @test logdet_jacobian(layer, [1.5, 6.9]) == 0.0
        @test logdet_jacobian(layer, [2.5, 6.4]) == 0.0
        @test logabsdet_jacobian(layer, [1.5, 6.9]) == 0.0
        @test logabsdet_jacobian(layer, [2.5, 6.4]) == 0.0

    end

end

end