module FlowNodeModelsFlowModelTest

using Test
using ReactiveMP
using ReactiveMP: getforward, getbackward, getjacobian, getinv_jacobian
using ReactiveMP: forward, forward!, backward, backward!, jacobian, jacobian!, inv_jacobian, inv_jacobian!, forward_jacobian, backward_inv_jacobian
using ReactiveMP: det_jacobian, absdet_jacobian, logdet_jacobian, logabsdet_jacobian, detinv_jacobian, absdetinv_jacobian, logabsdetinv_jacobian
@testset "Flow Model" begin
    @testset "Constructor" begin

        # check for single layer
        f = PlanarFlow()
        layer = AdditiveCouplingLayer(f; permute = false)
        model = FlowModel(2, (layer,))
        compiled_model = compile(model)
        @test length(compiled_model.layers) == 1
        @test typeof(compiled_model.layers[1]) <: AdditiveCouplingLayer
        @test typeof(compiled_model.layers[1].f[1]) <: PlanarFlow

        # check for two layers
        f1 = PlanarFlow()
        layer1 = AdditiveCouplingLayer(f1)
        f2 = PlanarFlow()
        layer2 = AdditiveCouplingLayer(f2; permute = false)
        model = FlowModel(2, (layer1, layer2))
        compiled_model = compile(model)
        @test length(compiled_model.layers) == 3
        @test typeof(compiled_model.layers[1]) <: AdditiveCouplingLayer
        @test typeof(compiled_model.layers[1].f[1]) <: PlanarFlow
        @test typeof(compiled_model.layers[2]) <: PermutationLayer
        @test typeof(compiled_model.layers[3]) <: AdditiveCouplingLayer
        @test typeof(compiled_model.layers[3].f[1]) <: PlanarFlow
    end

    @testset "Get" begin

        # check get functions for single layer model
        f = PlanarFlow()
        layer = AdditiveCouplingLayer(f; permute = false)
        model = FlowModel(2, (layer,))
        compiled_model = compile(model)
        @test getlayers(compiled_model) == compiled_model.layers
        @test length(getlayers(compiled_model)) == 1
        @test typeof(getforward(compiled_model)) <: Function
        @test typeof(getbackward(compiled_model)) <: Function
        @test typeof(getjacobian(compiled_model)) <: Function
        @test typeof(getinv_jacobian(compiled_model)) <: Function

        # check get functions for multi layer model
        f1 = PlanarFlow()
        layer1 = AdditiveCouplingLayer(f1)
        f2 = PlanarFlow()
        layer2 = AdditiveCouplingLayer(f2; permute = false)
        model = FlowModel(2, (layer1, layer2))
        compiled_model = compile(model)
        @test getlayers(compiled_model) == compiled_model.layers
        @test length(getlayers(compiled_model)) == 3
        @test typeof(getforward(compiled_model)) <: Function
        @test typeof(getbackward(compiled_model)) <: Function
        @test typeof(getjacobian(compiled_model)) <: Function
        @test typeof(getinv_jacobian(compiled_model)) <: Function
    end

    @testset "Base" begin

        # check base functions (single layer)
        f = PlanarFlow()
        layer = AdditiveCouplingLayer(f; permute = false)
        model = FlowModel(2, (layer,))
        compiled_model = compile(model)
        @test eltype(compiled_model) == Float64
        @test length(compiled_model) == 1

        # check base functions (multi layer)
        f1 = PlanarFlow()
        layer1 = AdditiveCouplingLayer(f1;)
        f2 = PlanarFlow()
        layer2 = AdditiveCouplingLayer(f2; permute = false)
        model = FlowModel(2, (layer1, layer2))
        compiled_model = compile(model)
        @test eltype(compiled_model) == Float64
        @test length(compiled_model) == 3
    end

    @testset "Forward-Backward" begin

        # check forward function (single layer)
        params = [1.0, 2.0, -3.0]
        f = PlanarFlow()
        layer = AdditiveCouplingLayer(f; permute = false)
        model = FlowModel(2, (layer,))
        compiled_model = compile(model, params)
        @test forward(compiled_model, [5.0, 1.5]) == [5.0, 7.4999983369439445]
        @test forward(compiled_model, [4.0, 2.5]) == [4.0, 7.499909204262595]
        @test forward.(compiled_model, [[5.0, 1.5], [4.0, 2.5]]) == [[5.0, 7.4999983369439445], [4.0, 7.499909204262595]]

        # check forward function (multiple layers)
        params = [1.0, 2.0, -3.0, 1.0, 2.0, -3.0]
        f1 = PlanarFlow()
        layer1 = AdditiveCouplingLayer(f1)
        f2 = PlanarFlow()
        layer2 = AdditiveCouplingLayer(f2; permute = false)
        model = FlowModel(2, (layer1, layer2))
        compiled_model = compile(model, params)
        @test forward(compiled_model, [5.0, 1.5]) == [7.4999983369439445, 13.49999833686844]
        @test forward(compiled_model, [4.0, 2.5]) == [7.499909204262595, 12.499909204187064]
        @test forward.(compiled_model, [[5.0, 1.5], [4.0, 2.5]]) == [[7.4999983369439445, 13.49999833686844], [7.499909204262595, 12.499909204187064]]

        # check forward! function (single layer)
        params = [1.0, 2.0, -3.0]
        f = PlanarFlow()
        layer = AdditiveCouplingLayer(f; permute = false)
        model = FlowModel(2, (layer,))
        compiled_model = compile(model, params)
        output = zeros(2)
        forward!(output, compiled_model, [5.0, 1.5])
        @test output == [5.0, 7.4999983369439445]
        forward!(output, compiled_model, [4.0, 2.5])
        @test output == [4.0, 7.499909204262595]

        # check forward! function (multiple layers)
        params = [1.0, 2.0, -3.0, 1.0, 2.0, -3.0]
        f1 = PlanarFlow()
        layer1 = AdditiveCouplingLayer(f1)
        f2 = PlanarFlow()
        layer2 = AdditiveCouplingLayer(f2; permute = false)
        model = FlowModel(2, (layer1, layer2))
        compiled_model = compile(model, params)
        output = zeros(2)
        forward!(output, compiled_model, [5.0, 1.5])
        @test output == [7.4999983369439445, 13.49999833686844]
        forward!(output, compiled_model, [4.0, 2.5])
        @test output == [7.499909204262595, 12.499909204187064]

        # check backward function (single layer)
        params = [1.0, 2.0, -3.0]
        f = PlanarFlow()
        layer = AdditiveCouplingLayer(f; permute = false)
        model = FlowModel(2, (layer,))
        compiled_model = compile(model, params)
        @test backward(compiled_model, [5.0, 7.4999983369439445]) == [5.0, 1.5]
        @test backward(compiled_model, [4.0, 7.499909204262595]) == [4.0, 2.5]
        @test backward.(compiled_model, [[5.0, 7.4999983369439445], [4.0, 7.499909204262595]]) == [[5.0, 1.5], [4.0, 2.5]]

        # check backward function (multiple layers)
        params = [1.0, 2.0, -3.0, 1.0, 2.0, -3.0]
        f1 = PlanarFlow()
        layer1 = AdditiveCouplingLayer(f1)
        f2 = PlanarFlow()
        layer2 = AdditiveCouplingLayer(f2; permute = false)
        model = FlowModel(2, (layer1, layer2))
        compiled_model = compile(model, params)
        @test backward(compiled_model, [7.4999983369439445, 13.49999833686844]) == [5.0, 1.5]
        @test backward(compiled_model, [7.499909204262595, 12.499909204187064]) == [4.0, 2.5]
        @test backward.(compiled_model, [[7.4999983369439445, 13.49999833686844], [7.499909204262595, 12.499909204187064]]) == [[5.0, 1.5], [4.0, 2.5]]

        # check backward! function (single layer)
        params = [1.0, 2.0, -3.0]
        f = PlanarFlow()
        layer = AdditiveCouplingLayer(f; permute = false)
        model = FlowModel(2, (layer,))
        compiled_model = compile(model, params)
        output = zeros(2)
        backward!(output, compiled_model, [5.0, 7.4999983369439445])
        @test output == [5.0, 1.5]
        backward!(output, compiled_model, [4.0, 7.499909204262595])
        @test output == [4.0, 2.5]

        # check backward! function (multiple layers)
        params = [1.0, 2.0, -3.0, 1.0, 2.0, -3.0]
        f1 = PlanarFlow()
        layer1 = AdditiveCouplingLayer(f1)
        f2 = PlanarFlow()
        layer2 = AdditiveCouplingLayer(f2; permute = false)
        model = FlowModel(2, (layer1, layer2))
        compiled_model = compile(model, params)
        output = zeros(2)
        backward!(output, compiled_model, [7.4999983369439445, 13.49999833686844])
        @test output == [5.0, 1.5]
        backward!(output, compiled_model, [7.499909204262595, 12.499909204187064])
        @test output == [4.0, 2.5]
    end

    @testset "Jacobian" begin

        # check jacobian function (single layer)
        params = [1.0, 2.0, -3.0]
        f = PlanarFlow()
        layer = AdditiveCouplingLayer(f; permute = false)
        model = FlowModel(2, (layer,))
        compiled_model = compile(model, params)
        @test jacobian(compiled_model, [3.0, 1.5]) == [1.0 0.0; 1.0197320743308804 1.0]
        @test jacobian(compiled_model, [2.5, 5.0]) == [1.0 0.0; 1.1413016497063289 1.0]
        @test jacobian.(compiled_model, [[3.0, 1.5], [2.5, 5.0]]) == [[1.0 0.0; 1.0197320743308804 1.0], [1.0 0.0; 1.1413016497063289 1.0]]

        # check jacobian! function (single layer)
        params = [1.0, 2.0, -3.0]
        f = PlanarFlow()
        layer = AdditiveCouplingLayer(f; permute = false)
        model = FlowModel(2, (layer,))
        compiled_model = compile(model, params)
        output = zeros(2, 2)
        jacobian!(output, compiled_model, [3.0, 1.5])
        @test output == [1.0 0.0; 1.0197320743308804 1.0]
        jacobian!(output, compiled_model, [2.5, 5.0])
        @test output == [1.0 0.0; 1.1413016497063289 1.0]

        # check jacobian function (multiple layers)
        params = [1.0, 2.0, -3.0, 1.0, 2.0, -3.0]
        f1 = PlanarFlow()
        layer1 = AdditiveCouplingLayer(f1)
        f2 = PlanarFlow()
        layer2 = AdditiveCouplingLayer(f2; permute = false)
        model = FlowModel(2, (layer1, layer2))
        compiled_model = compile(model, params)
        @test jacobian(compiled_model, [3.0, 1.5]) == [1.0197320743308804 1.0; 2.0197330107171334 1.0000009182669414]
        @test jacobian(compiled_model, [2.5, 5.0]) == [1.1413016497063289 1.0; 2.1413016497136192 1.0000000000063878]
        @test jacobian.(compiled_model, [[3.0, 1.5], [2.5, 5.0]]) ==
            [[1.0197320743308804 1.0; 2.0197330107171334 1.0000009182669414], [1.1413016497063289 1.0; 2.1413016497136192 1.0000000000063878]]

        # check jacobian! function (multiple layers)
        params = [1.0, 2.0, -3.0, 1.0, 2.0, -3.0]
        f1 = PlanarFlow()
        layer1 = AdditiveCouplingLayer(f1)
        f2 = PlanarFlow()
        layer2 = AdditiveCouplingLayer(f2; permute = false)
        model = FlowModel(2, (layer1, layer2))
        compiled_model = compile(model, params)
        output = zeros(2, 2)
        jacobian!(output, compiled_model, [3.0, 1.5])
        @test output == [1.0197320743308804 1.0; 2.0197330107171334 1.0000009182669414]
        jacobian!(output, compiled_model, [2.5, 5.0])
        @test output == [1.1413016497063289 1.0; 2.1413016497136192 1.0000000000063878]

        # check inv_jacobian function (single layer)
        params = [1.0, 2.0, -3.0]
        f = PlanarFlow()
        layer = AdditiveCouplingLayer(f; permute = false)
        model = FlowModel(2, (layer,))
        compiled_model = compile(model, params)
        @test inv_jacobian(compiled_model, [3.0, 1.5]) == [1.0 0.0; -1.0197320743308804 1.0]
        @test inv_jacobian(compiled_model, [2.5, 5.0]) == [1.0 0.0; -1.1413016497063289 1.0]
        @test inv_jacobian.(compiled_model, [[3.0, 1.5], [2.5, 5.0]]) == [[1.0 0.0; -1.0197320743308804 1.0], [1.0 0.0; -1.1413016497063289 1.0]]

        # check inv_jacobian! function (single layer)
        params = [1.0, 2.0, -3.0]
        f = PlanarFlow()
        layer = AdditiveCouplingLayer(f; permute = false)
        model = FlowModel(2, (layer,))
        compiled_model = compile(model, params)
        output = zeros(2, 2)
        inv_jacobian!(output, compiled_model, [3.0, 1.5])
        @test output == [1.0 0.0; -1.0197320743308804 1.0]
        inv_jacobian!(output, compiled_model, [2.5, 5.0])
        @test output == [1.0 0.0; -1.1413016497063289 1.0]

        # check inv_jacobian! function (multiple layers)
        params = [1.0, 2.0, -3.0, 1.0, 2.0, -3.0]
        f1 = PlanarFlow()
        layer1 = AdditiveCouplingLayer(f1)
        f2 = PlanarFlow()
        layer2 = AdditiveCouplingLayer(f2; permute = false)
        model = FlowModel(2, (layer1, layer2))
        compiled_model = compile(model, params)
        output = zeros(2, 2)
        inv_jacobian!(output, compiled_model, [3.0, 1.5])
        @test output == [-1.0197320743308804 1.0; 2.0197330107171334 -1.0000009182669414]
        inv_jacobian!(output, compiled_model, [2.5, 5.0])
        @test output == [-1.1413016497063289 1.0; 4.412130707993816 -2.9896834976728544]

        # check inv_jacobian function (multiple layers)
        params = [1.0, 2.0, -3.0, 1.0, 2.0, -3.0]
        f1 = PlanarFlow()
        layer1 = AdditiveCouplingLayer(f1)
        f2 = PlanarFlow()
        layer2 = AdditiveCouplingLayer(f2; permute = false)
        model = FlowModel(2, (layer1, layer2))
        compiled_model = compile(model, params)
        @test inv_jacobian(compiled_model, [3.0, 1.5]) == [-1.0197320743308804 1.0; 2.0197330107171334 -1.0000009182669414]
        @test inv_jacobian(compiled_model, [2.5, 5.0]) == [-1.1413016497063289 1.0; 4.412130707993816 -2.9896834976728544]
        @test inv_jacobian.(compiled_model, [[3.0, 1.5], [2.5, 5.0]]) ==
            [[-1.0197320743308804 1.0; 2.0197330107171334 -1.0000009182669414], [-1.1413016497063289 1.0; 4.412130707993816 -2.9896834976728544]]
    end

    @testset "Joint processing functions" begin
        model = FlowModel((InputLayer(8), AdditiveCouplingLayer(PlanarFlow()), AdditiveCouplingLayer(PlanarFlow(); permute = false)))
        compiled_model = compile(model)
        x = randn(8)
        @test forward_jacobian(compiled_model, x) == (forward(compiled_model, x), jacobian(compiled_model, x))
        @test backward_inv_jacobian(compiled_model, x) == (backward(compiled_model, x), inv_jacobian(compiled_model, x))
        @test inv(jacobian(compiled_model, x)) ≈ inv_jacobian(compiled_model, forward(compiled_model, x))
    end

    @testset "Utility Jacobian" begin

        # check utility functions jacobian (single layer)
        params = [1.0, 2.0, -3.0]
        f = PlanarFlow()
        layer = AdditiveCouplingLayer(f; permute = false)
        model = FlowModel(2, (layer,))
        compiled_model = compile(model, params)
        @test det_jacobian(compiled_model, [1.5, 6.9]) == 1.0
        @test det_jacobian(compiled_model, [2.5, 6.4]) == 1.0
        @test absdet_jacobian(compiled_model, [1.5, 6.9]) == 1.0
        @test absdet_jacobian(compiled_model, [2.5, 6.4]) == 1.0
        @test isapprox(logdet_jacobian(compiled_model, [1.5, 6.9]), 0.0; atol = 1e-10)
        @test isapprox(logdet_jacobian(compiled_model, [2.5, 6.4]), 0.0; atol = 1e-10)
        @test sum(isapprox.(logabsdet_jacobian(compiled_model, [1.5, 6.9]), (0.0, 1.0); atol = 1e-10)) == 2
        @test sum(isapprox.(logabsdet_jacobian(compiled_model, [2.5, 6.4]), (0.0, 1.0); atol = 1e-10)) == 2

        # check utility functions jacobian (multiple layers)
        params = [1.0, 2.0, -3.0, 1.0, 2.0, -3.0]
        f1 = PlanarFlow()
        layer1 = AdditiveCouplingLayer(f1)
        f2 = PlanarFlow()
        layer2 = AdditiveCouplingLayer(f2; permute = false)
        model = FlowModel(2, (layer1, layer2))
        compiled_model = compile(model, params)
        @test detinv_jacobian(compiled_model, [1.5, 6.9]) ≈ -1.0
        @test detinv_jacobian(compiled_model, [2.5, 6.4]) ≈ -1.0
        @test absdetinv_jacobian(compiled_model, [1.5, 6.9]) ≈ 1.0
        @test absdetinv_jacobian(compiled_model, [2.5, 6.4]) ≈ 1.0
        @test sum(isapprox.(logabsdetinv_jacobian(compiled_model, [1.5, 6.9]), (0.0, -1.0); atol = 1e-10)) == 2
        @test sum(isapprox.(logabsdetinv_jacobian(compiled_model, [2.5, 6.4]), (0.0, -1.0); atol = 1e-10)) == 2
    end
end

end
