
@testitem "Additive Coupling Layer" begin
    using ReactiveMP
    using ReactiveMP: getf, getflow, getdim, forward, forward!, backward, backward!, jacobian, inv_jacobian
    using ReactiveMP: det_jacobian, absdet_jacobian, logdet_jacobian, logabsdet_jacobian

    @testset "Constructor" begin

        # check for standard layer
        f = PlanarFlow()
        layer = AdditiveCouplingLayer(f)
        @test typeof(layer.f) <: ReactiveMP.PlanarFlowEmpty
        @test typeof(layer) <: ReactiveMP.AdditiveCouplingLayerPlaceholder
        @test layer.partition_dim == 1
        @test layer.permute == Val(true)

        f = PlanarFlow()
        layer = AdditiveCouplingLayer(f; permute = false, partition_dim = 3)
        @test typeof(layer.f) <: ReactiveMP.PlanarFlowEmpty
        @test typeof(layer) <: ReactiveMP.AdditiveCouplingLayerPlaceholder
        @test layer.partition_dim == 3
        @test layer.permute == Val(false)
    end

    @testset "Prepare-Compile" begin
        layert = AdditiveCouplingLayer(PlanarFlow())
        layerf = AdditiveCouplingLayer(PlanarFlow(); permute = false)
        outt = ReactiveMP._prepare(2, layert)
        outf = ReactiveMP._prepare(2, layerf)

        @test typeof(outt[1]) <: ReactiveMP.AdditiveCouplingLayerEmpty
        @test typeof(outt[2]) <: ReactiveMP.PermutationLayer
        @test typeof(outf) <: ReactiveMP.AdditiveCouplingLayerEmpty

        @test outt[1].dim == 2
        @test outt[2].dim == 2
        @test outf.dim == 2

        layer_comp  = compile(outf)
        layer_compp = compile(outf, [1.0, 2.0, 3.0])

        @test typeof(layer_comp) <: AdditiveCouplingLayer
        @test typeof(layer_compp) <: AdditiveCouplingLayer
        @test typeof(layer_comp.f) <: Tuple
        @test typeof(layer_comp.f[1]) <: PlanarFlow
        @test typeof(layer_compp.f) <: Tuple
        @test typeof(layer_compp.f[1]) <: PlanarFlow
        @test layer_compp.f[1].u == 1.0
        @test layer_compp.f[1].w == 2.0
        @test layer_compp.f[1].b == 3.0

        # TODO expend for multiple mappings
        @test nr_params(outf) == 3
        @test nr_params(layer_comp) == 3
    end

    @testset "Get" begin

        # check get functions for univariate PlanarFlow
        f = PlanarFlow()
        layer = AdditiveCouplingLayer(f; permute = false)
        out = compile(ReactiveMP._prepare(2, layer))
        @test getf(out) == out.f
        @test getflow(out) == out.f
        @test getdim(out) == out.dim
        @test getdim(out) == 2
    end

    @testset "Base" begin

        # check base functions (univariate)
        f = PlanarFlow()
        layer = AdditiveCouplingLayer(f; permute = false)
        out = compile(ReactiveMP._prepare(2, layer))
        @test eltype(out) == Float64
    end

    @testset "Forward-Backward" begin

        # check forward function
        params = [1.0, 2.0, -3.0]
        f = PlanarFlow()
        layer = AdditiveCouplingLayer(f; permute = false)
        layer = compile(ReactiveMP._prepare(2, layer), params)
        @test forward(layer, [5.0, 1.5]) == [5.0, 7.4999983369439445]
        @test forward(layer, [4.0, 2.5]) == [4.0, 7.499909204262595]
        @test forward.(layer, [[5.0, 1.5], [4.0, 2.5]]) == [[5.0, 7.4999983369439445], [4.0, 7.499909204262595]]

        # check forward! function
        params = [1.0, 2.0, -3.0]
        f = PlanarFlow()
        layer = AdditiveCouplingLayer(f; permute = false)
        layer = compile(ReactiveMP._prepare(2, layer), params)
        output = zeros(2)
        forward!(output, layer, [5.0, 1.5])
        @test output == [5.0, 7.4999983369439445]
        forward!(output, layer, [4.0, 2.5])
        @test output == [4.0, 7.499909204262595]

        # check forward function (input > 2)
        f = PlanarFlow()
        layer = AdditiveCouplingLayer(f; permute = false)
        layer = compile(ReactiveMP._prepare(3, layer))
        x = randn(3)
        @test backward(layer, forward(layer, x)) ≈ x

        # check backward function
        params = [1.0, 2.0, -3.0]
        f = PlanarFlow()
        layer = AdditiveCouplingLayer(f; permute = false)
        layer = compile(ReactiveMP._prepare(2, layer), params)
        @test backward(layer, [5.0, 7.4999983369439445]) == [5.0, 1.5]
        @test backward(layer, [4.0, 7.499909204262595]) == [4.0, 2.5]
        @test backward.(layer, [[5.0, 7.4999983369439445], [4.0, 7.499909204262595]]) == [[5.0, 1.5], [4.0, 2.5]]

        # check backward! function
        params = [1.0, 2.0, -3.0]
        f = PlanarFlow()
        layer = AdditiveCouplingLayer(f; permute = false)
        layer = compile(ReactiveMP._prepare(2, layer), params)
        output = zeros(2)
        backward!(output, layer, [5.0, 7.4999983369439445])
        @test output == [5.0, 1.5]
        backward!(output, layer, [4.0, 7.499909204262595])
        @test output == [4.0, 2.5]
    end

    @testset "Jacobian" begin

        # check jacobian function
        params = [1.0, 2.0, -3.0]
        f = PlanarFlow()
        layer = AdditiveCouplingLayer(f; permute = false)
        layer = compile(ReactiveMP._prepare(2, layer), params)
        @test jacobian(layer, [3.0, 1.5]) == [1.0 0.0; 1.0197320743308804 1.0]
        @test jacobian(layer, [2.5, 5.0]) == [1.0 0.0; 1.1413016497063289 1.0]
        @test jacobian.(layer, [[3.0, 1.5], [2.5, 5.0]]) == [[1.0 0.0; 1.0197320743308804 1.0], [1.0 0.0; 1.1413016497063289 1.0]]

        # check jacobian function
        params = [1.0, 2.0, -3.0]
        f = PlanarFlow()
        layer = AdditiveCouplingLayer(f; permute = false)
        layer = compile(ReactiveMP._prepare(2, layer), params)
        @test inv_jacobian(layer, [3.0, 1.5]) == [1.0 0.0; -1.0197320743308804 1.0]
        @test inv_jacobian(layer, [2.5, 5.0]) == [1.0 0.0; -1.1413016497063289 1.0]
        @test inv_jacobian.(layer, [[3.0, 1.5], [2.5, 5.0]]) == [[1.0 0.0; -1.0197320743308804 1.0], [1.0 0.0; -1.1413016497063289 1.0]]

        # check for invertibility 
        layer = AdditiveCouplingLayer(PlanarFlow(); permute = false)
        x = randn(10)
        layer = compile(ReactiveMP._prepare(10, layer))
        @test inv(jacobian(layer, x)) ≈ inv_jacobian(layer, forward(layer, x))
    end

    @testset "Utility Jacobian" begin

        # check utility functions jacobian (univariate)
        params = [1.0, 2.0, -3.0]
        f = PlanarFlow()
        layer = AdditiveCouplingLayer(f; permute = false)
        layer = compile(ReactiveMP._prepare(2, layer), params)
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
