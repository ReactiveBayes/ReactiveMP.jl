module FlowNodeLayersNiceLayerTest

using Test
using ReactiveMP 

@testset "Nice Layer" begin

    @testset "Constructor" begin
        
        # check for layer with univariate function
        f = PlanarMap()
        layer = NiceLayer(f)
        @test layer.f == f
        @test layer.f.u == f.u
        @test layer.f.w == f.w
        @test layer.f.b == f.b

        # check for layer with multivariate function
        f = PlanarMap(3)
        layer = NiceLayer(f)
        @test layer.f == f
        @test layer.f.u == f.u
        @test layer.f.w == f.w
        @test layer.f.b == f.b

    end

    @testset "Get" begin
        
        # check get functions for univariate PlanarMap
        f = PlanarMap()
        layer = NiceLayer(f)
        @test getf(layer) == layer.f
        @test getmap(layer) == layer.f
        @test getf(layer) == f
        @test getmap(layer) == f

        # check get functions for multivariate PlanarMap
        f = PlanarMap(3)
        layer = NiceLayer(f)
        @test getf(layer) == layer.f
        @test getmap(layer) == layer.f
        @test getf(layer) == f
        @test getmap(layer) == f

    end

    @testset "Base" begin
        
        # check base functions (univariate)
        f = PlanarMap()
        layer = NiceLayer(f)
        @test eltype(layer) == Float64
        @test eltype(NiceLayer{PlanarMap{Float64,Float64}}) == Float64

        # check base functions (multivariate)
        f = PlanarMap(3)
        layer = NiceLayer(f)
        @test eltype(layer) == Float64
        @test eltype(NiceLayer{PlanarMap{Array{Float64,1},Float64}}) == Float64

    end

    @testset "Forward-Backward" begin
        
        # check forward function
        f = PlanarMap(1.0, 2.0, -3.0)
        layer = NiceLayer(f)
        @test forward(layer, [5.0, 1.5]) == [5.0, 7.4999983369439445]
        @test forward(layer, [4.0, 2.5]) == [4.0, 7.499909204262595]
        @test forward.(layer, [[5.0, 1.5], [4.0, 2.5]]) == [[5.0, 7.4999983369439445], [4.0, 7.499909204262595]]

        # check forward! function
        f = PlanarMap(1.0, 2.0, -3.0)
        layer = NiceLayer(f)
        output = zeros(2)
        forward!(output, layer, [5.0, 1.5]) 
        @test output == [5.0, 7.4999983369439445]
        forward!(output, layer, [4.0, 2.5]) 
        @test output == [4.0, 7.499909204262595]

        # check backward function
        f = PlanarMap(1.0, 2.0, -3.0)
        layer = NiceLayer(f)
        @test backward(layer, [5.0, 7.4999983369439445]) == [5.0, 1.5]
        @test backward(layer, [4.0, 7.499909204262595]) == [4.0, 2.5]
        @test backward.(layer, [[5.0, 7.4999983369439445], [4.0, 7.499909204262595]]) == [[5.0, 1.5], [4.0, 2.5]]

        # check backward! function
        f = PlanarMap(1.0, 2.0, -3.0)
        layer = NiceLayer(f)
        output = zeros(2)
        backward!(output, layer, [5.0, 7.4999983369439445]) 
        @test output == [5.0, 1.5]
        backward!(output, layer, [4.0, 7.499909204262595]) 
        @test output == [4.0, 2.5]

    end

    @testset "Jacobian" begin
        
        # check jacobian function
        f = PlanarMap(1.0, 2.0, -3.0)
        layer = NiceLayer(f)
        @test jacobian(layer, [3.0, 1.5]) == [1.0 0.0; 1.0197320743308804 1.0]
        @test jacobian(layer, [2.5, 5.0]) == [1.0 0.0; 1.1413016497063289 1.0]
        @test jacobian.(layer, [[3.0, 1.5], [2.5, 5.0]]) == [[1.0 0.0; 1.0197320743308804 1.0], [1.0 0.0; 1.1413016497063289 1.0]]

        # check jacobian function
        f = PlanarMap(1.0, 2.0, -3.0)
        layer = NiceLayer(f)
        @test inv_jacobian(layer, [3.0, 1.5]) == [1.0 0.0; -1.0197320743308804 1.0]
        @test inv_jacobian(layer, [2.5, 5.0]) == [1.0 0.0; -1.1413016497063289 1.0]
        @test inv_jacobian.(layer, [[3.0, 1.5], [2.5, 5.0]]) == [[1.0 0.0; -1.0197320743308804 1.0], [1.0 0.0; -1.1413016497063289 1.0]]        

    end

    @testset "Utility Jacobian" begin
        
        # check utility functions jacobian (univariate)
        f = PlanarMap(1.0, 2.0, -3.0)
        layer = NiceLayer(f)
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