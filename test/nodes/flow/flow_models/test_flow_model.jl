module FlowNodeModelsFlowModelTest

using Test
using ReactiveMP 

@testset "Flow Model" begin

    @testset "Constructor" begin
        
        # check for single layer
        f = PlanarMap()
        layer = NiceLayer(f)
        model = FlowModel( (layer, ) )
        @test model.layers == (layer, )
        @test model.layers[1].f == f
        @test model.layers[1].f.u == f.u
        @test model.layers[1].f.w == f.w
        @test model.layers[1].f.b == f.b

        # check for two layers
        f1 = PlanarMap()
        layer1 = NiceLayer(f1)
        f2 = PlanarMap()
        layer2 = ReverseNiceLayer(f2)
        model = FlowModel( (layer1, layer2) )
        @test model.layers == (layer1, layer2)
        @test model.layers[1] == layer1
        @test model.layers[2] == layer2
        @test model.layers[1].f == f1
        @test model.layers[1].f.u == f1.u
        @test model.layers[1].f.w == f1.w
        @test model.layers[1].f.b == f1.b
        @test model.layers[2].f == f2
        @test model.layers[2].f.u == f2.u
        @test model.layers[2].f.w == f2.w
        @test model.layers[2].f.b == f2.b

    end

    @testset "Get" begin
        
        # check get functions for single layer model
        f = PlanarMap()
        layer = NiceLayer(f)
        model = FlowModel( (layer, ) )
        @test getlayers(model) == model.layers
        @test getlayers(model) == (layer, )
        @test typeof(getforward(model)) <: Function
        @test typeof(getbackward(model)) <: Function
        @test typeof(getjacobian(model)) <: Function
        @test typeof(getinv_jacobian(model)) <: Function

        # check get functions for multi layer model
        f1 = PlanarMap()
        layer1 = NiceLayer(f1)
        f2 = PlanarMap()
        layer2 = ReverseNiceLayer(f2)
        model = FlowModel( (layer1, layer2) )
        @test getlayers(model) == model.layers
        @test getlayers(model) == (layer1, layer2)
        @test typeof(getforward(model)) <: Function
        @test typeof(getbackward(model)) <: Function
        @test typeof(getjacobian(model)) <: Function
        @test typeof(getinv_jacobian(model)) <: Function

    end

    @testset "Base" begin
        
        # check base functions (single layer)
        f = PlanarMap()
        layer = NiceLayer(f)
        model = FlowModel( (layer, ) )
        @test eltype(model) == Float64
        @test length(model) == 1


        # check base functions (multi layer)
        f1 = PlanarMap()
        layer1 = NiceLayer(f1)
        f2 = PlanarMap()
        layer2 = ReverseNiceLayer(f2)
        model = FlowModel( (layer1, layer2) )
        @test eltype(model) == Float64
        @test length(model) == 2

    end

    @testset "Forward-Backward" begin
        
        # check forward function (single layer)
        f = PlanarMap(1.0, 2.0, -3.0)
        layer = NiceLayer(f)
        model = FlowModel( (layer, ) )
        @test forward(model, [5.0, 1.5]) == [5.0, 7.4999983369439445]
        @test forward(model, [4.0, 2.5]) == [4.0, 7.499909204262595] 
        @test forward.(model, [[5.0, 1.5], [4.0, 2.5]]) == [[5.0, 7.4999983369439445], [4.0, 7.499909204262595] ]

        # check forward function (multiple layers)
        f1 = PlanarMap(1.0, 2.0, -3.0)
        layer1 = NiceLayer(f1)
        f2 = PlanarMap(1.0, 2.0, -3.0)
        layer2 = ReverseNiceLayer(f2)
        model = FlowModel( (layer1, layer2) )
        @test forward(model, [5.0, 1.5]) == [13.49999833686844, 7.4999983369439445]
        @test forward(model, [4.0, 2.5]) == [12.499909204187064, 7.499909204262595]
        @test forward.(model, [[5.0, 1.5], [4.0, 2.5]]) == [[13.49999833686844, 7.4999983369439445], [12.499909204187064, 7.499909204262595]]

        # check forward! function (single layer)
        f = PlanarMap(1.0, 2.0, -3.0)
        layer = NiceLayer(f)
        model = FlowModel( (layer, ) )
        output = zeros(2)
        forward!(output, model, [5.0, 1.5]) 
        @test output == [5.0, 7.4999983369439445]
        forward!(output, model, [4.0, 2.5]) 
        @test output == [4.0, 7.499909204262595]

        # check forward! function (multiple layers)
        f1 = PlanarMap(1.0, 2.0, -3.0)
        layer1 = NiceLayer(f1)
        f2 = PlanarMap(1.0, 2.0, -3.0)
        layer2 = ReverseNiceLayer(f2)
        model = FlowModel( (layer1, layer2) )
        output = zeros(2)
        forward!(output, model, [5.0, 1.5]) 
        @test output == [13.49999833686844, 7.4999983369439445]
        forward!(output, model, [4.0, 2.5]) 
        @test output == [12.499909204187064, 7.499909204262595]

        # check backward function (single layer)
        f = PlanarMap(1.0, 2.0, -3.0)
        layer = NiceLayer(f)
        model = FlowModel( (layer, ) )
        @test backward(model, [5.0, 7.4999983369439445]) == [5.0, 1.5]
        @test backward(model, [4.0, 7.499909204262595]) == [4.0, 2.5]
        @test backward.(model, [[5.0, 7.4999983369439445], [4.0, 7.499909204262595]]) == [[5.0, 1.5], [4.0, 2.5]]

        # check backward function (multiple layers)
        f1 = PlanarMap(1.0, 2.0, -3.0)
        layer1 = NiceLayer(f1)
        f2 = PlanarMap(1.0, 2.0, -3.0)
        layer2 = ReverseNiceLayer(f2)
        model = FlowModel( (layer1, layer2) )
        @test backward(model, [13.49999833686844, 7.4999983369439445]) == [5.0, 1.5] 
        @test backward(model, [12.499909204187064, 7.499909204262595]) == [4.0, 2.5] 
        @test backward.(model, [[13.49999833686844, 7.4999983369439445], [12.499909204187064, 7.499909204262595]]) == [[5.0, 1.5], [4.0, 2.5]]

        # check backward! function (single layer)
        f = PlanarMap(1.0, 2.0, -3.0)
        layer = NiceLayer(f)
        model = FlowModel( (layer, ) )
        output = zeros(2)
        backward!(output, model, [5.0, 7.4999983369439445]) 
        @test output == [5.0, 1.5]
        backward!(output, model, [4.0, 7.499909204262595]) 
        @test output == [4.0, 2.5]

        # check backward! function (multiple layers)
        f1 = PlanarMap(1.0, 2.0, -3.0)
        layer1 = NiceLayer(f1)
        f2 = PlanarMap(1.0, 2.0, -3.0)
        layer2 = ReverseNiceLayer(f2)
        model = FlowModel( (layer1, layer2) )
        output = zeros(2)
        backward!(output, model, [13.49999833686844, 7.4999983369439445]) 
        @test output == [5.0, 1.5]
        backward!(output, model, [12.499909204187064, 7.499909204262595]) 
        @test output == [4.0, 2.5]

    end

    @testset "Jacobian" begin
        
        # check jacobian function (single layer)
        f = PlanarMap(1.0, 2.0, -3.0)
        layer = NiceLayer(f)
        model = FlowModel( (layer, ) )
        @test jacobian(model, [3.0, 1.5]) == [1.0 0.0; 1.0197320743308804 1.0]
        @test jacobian(model, [2.5, 5.0]) == [1.0 0.0; 1.1413016497063289 1.0]
        @test jacobian.(model, [[3.0, 1.5], [2.5, 5.0]]) == [[1.0 0.0; 1.0197320743308804 1.0], [1.0 0.0; 1.1413016497063289 1.0]]

        # check jacobian! function (single layer)
        f = PlanarMap(1.0, 2.0, -3.0)
        layer = NiceLayer(f)
        model = FlowModel( (layer, ) )
        output = zeros(2,2)
        jacobian!(output, model, [3.0, 1.5]) 
        @test output == [1.0 0.0; 1.0197320743308804 1.0]
        jacobian!(output, model, [2.5, 5.0]) 
        @test output == [1.0 0.0; 1.1413016497063289 1.0]

        # check jacobian function (multiple layers)
        f1 = PlanarMap(1.0, 2.0, -3.0)
        layer1 = NiceLayer(f1)
        f2 = PlanarMap(1.0, 2.0, -3.0)
        layer2 = ReverseNiceLayer(f2)
        model = FlowModel( (layer1, layer2) )
        @test jacobian(model, [3.0, 1.5]) == [2.0398535034191605 1.0197320743308804; 1.0197320743308804 1.0]
        @test jacobian(model, [2.5, 5.0]) == [2.302569455622388 1.1413016497063289; 1.1413016497063289 1.0]
        @test jacobian.(model, [[3.0, 1.5], [2.5, 5.0]]) == [[2.0398535034191605 1.0197320743308804; 1.0197320743308804 1.0], [2.302569455622388 1.1413016497063289; 1.1413016497063289 1.0]]

        # check jacobian! function (multiple layers)
        f1 = PlanarMap(1.0, 2.0, -3.0)
        layer1 = NiceLayer(f1)
        f2 = PlanarMap(1.0, 2.0, -3.0)
        layer2 = ReverseNiceLayer(f2)
        model = FlowModel( (layer1, layer2) )
        output = zeros(2,2)
        jacobian!(output, model, [3.0, 1.5]) 
        @test output == [2.0398535034191605 1.0197320743308804; 1.0197320743308804 1.0]
        jacobian!(output, model, [2.5, 5.0]) 
        @test output == [2.302569455622388 1.1413016497063289; 1.1413016497063289 1.0]

        # check inv_jacobian function (single layer)
        f = PlanarMap(1.0, 2.0, -3.0)
        layer = NiceLayer(f)
        model = FlowModel( (layer, ) )
        @test inv_jacobian(model, [3.0, 1.5]) == [1.0 0.0; -1.0197320743308804 1.0]
        @test inv_jacobian(model, [2.5, 5.0]) == [1.0 0.0; -1.1413016497063289 1.0]
        @test inv_jacobian.(model, [[3.0, 1.5], [2.5, 5.0]]) == [[1.0 0.0; -1.0197320743308804 1.0], [1.0 0.0; -1.1413016497063289 1.0]]        
        
        # check inv_jacobian! function (single layer)
        f = PlanarMap(1.0, 2.0, -3.0)
        layer = NiceLayer(f)
        model = FlowModel( (layer, ) )
        output = zeros(2,2)
        inv_jacobian!(output, model, [3.0, 1.5]) 
        @test output == [1.0 0.0; -1.0197320743308804 1.0]
        inv_jacobian!(output, model, [2.5, 5.0]) 
        @test output == [1.0 0.0; -1.1413016497063289 1.0]

        # check inv_jacobian! function (multiple layers)
        f1 = PlanarMap(1.0, 2.0, -3.0)
        layer1 = NiceLayer(f1)
        f2 = PlanarMap(1.0, 2.0, -3.0)
        layer2 = ReverseNiceLayer(f2)
        model = FlowModel( (layer1, layer2) )
        output = zeros(2,2)
        inv_jacobian!(output, model, [3.0, 1.5]) 
        @test output == [1.0 -1.0197320743308804; -3.0 4.059196222992641]
        inv_jacobian!(output, model, [2.5, 5.0]) 
        @test output == [1.0 -1.1413016497063289; -1.0000000164893388 2.1413016685256387]

        # check inv_jacobian function (multiple layers)
        f1 = PlanarMap(1.0, 2.0, -3.0)
        layer1 = NiceLayer(f1)
        f2 = PlanarMap(1.0, 2.0, -3.0)
        layer2 = ReverseNiceLayer(f2)
        model = FlowModel( (layer1, layer2) )
        @test inv_jacobian(model, [3.0, 1.5]) == [1.0 -1.0197320743308804; -3.0 4.059196222992641]
        @test inv_jacobian(model, [2.5, 5.0]) == [1.0 -1.1413016497063289; -1.0000000164893388 2.1413016685256387]
        @test inv_jacobian.(model, [[3.0, 1.5], [2.5, 5.0]]) == [[1.0 -1.0197320743308804; -3.0 4.059196222992641], [1.0 -1.1413016497063289; -1.0000000164893388 2.1413016685256387]]

    end

    @testset "Utility Jacobian" begin
        
        # check utility functions jacobian (single layer)
        f = PlanarMap(1.0, 2.0, -3.0)
        layer = NiceLayer(f)
        model = FlowModel( (layer, ) )
        @test det_jacobian(model, [1.5, 6.9]) == 1.0
        @test det_jacobian(model, [2.5, 6.4]) == 1.0
        @test absdet_jacobian(model, [1.5, 6.9]) == 1.0
        @test absdet_jacobian(model, [2.5, 6.4]) == 1.0
        @test isapprox(logdet_jacobian(model, [1.5, 6.9]), 0.0; atol=1e-10)
        @test isapprox(logdet_jacobian(model, [2.5, 6.4]), 0.0; atol=1e-10)
        @test sum(isapprox.(logabsdet_jacobian(model, [1.5, 6.9]), (0.0, 1.0); atol=1e-10)) == 2
        @test sum(isapprox.(logabsdet_jacobian(model, [2.5, 6.4]), (0.0, 1.0); atol=1e-10)) == 2

        # check utility functions jacobian (multiple layers)
        f1 = PlanarMap(1.0, 2.0, -3.0)
        layer1 = NiceLayer(f1)
        f2 = PlanarMap(1.0, 2.0, -3.0)
        layer2 = ReverseNiceLayer(f2)
        model = FlowModel( (layer1, layer2) )
        @test detinv_jacobian(model, [1.5, 6.9]) ≈ 1.0
        @test detinv_jacobian(model, [2.5, 6.4]) ≈ 1.0
        @test absdetinv_jacobian(model, [1.5, 6.9]) ≈ 1.0
        @test absdetinv_jacobian(model, [2.5, 6.4]) ≈ 1.0
        @test isapprox(logdetinv_jacobian(model, [1.5, 6.9]), 0.0; atol=1e-10)
        @test isapprox(logdetinv_jacobian(model, [2.5, 6.4]), 0.0; atol=1e-10)
        @test sum(isapprox.(logabsdetinv_jacobian(model, [1.5, 6.9]), (0.0, 1.0); atol=1e-10)) == 2
        @test sum(isapprox.(logabsdetinv_jacobian(model, [2.5, 6.4]), (0.0, 1.0); atol=1e-10)) == 2

    end

end

end