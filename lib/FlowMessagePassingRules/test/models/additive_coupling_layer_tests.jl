@testitem "models:Additive Coupling Layer" tags = [:models] begin
    using FlowMessagePassingRules, StableRNGs
    using FlowMessagePassingRules:
        getf,
        getflow,
        getdim,
        forward,
        forward!,
        backward,
        backward!,
        jacobian,
        inv_jacobian
    using FlowMessagePassingRules:
        det_jacobian, absdet_jacobian, logdet_jacobian, logabsdet_jacobian
    rng = StableRNG(1)

    @testset "Constructor" begin

        # check for standard layer
        f = PlanarFlow()
        layer = AdditiveCouplingLayer(f)
        @test typeof(layer.f) <: FlowMessagePassingRules.PlanarFlowEmpty
        @test typeof(layer) <: FlowMessagePassingRules.AdditiveCouplingLayerPlaceholder
        @test layer.partition_dim == 1
        @test layer.permute == Val(true)

        f = PlanarFlow()
        layer = AdditiveCouplingLayer(f; permute = false, partition_dim = 3)
        @test typeof(layer.f) <: FlowMessagePassingRules.PlanarFlowEmpty
        @test typeof(layer) <: FlowMessagePassingRules.AdditiveCouplingLayerPlaceholder
        @test layer.partition_dim == 3
        @test layer.permute == Val(false)
    end

    @testset "Prepare-Compile" begin
        layert = AdditiveCouplingLayer(PlanarFlow())
        layerf = AdditiveCouplingLayer(PlanarFlow(); permute = false)
        outt = FlowMessagePassingRules._prepare(rng, 2, layert)
        outf = FlowMessagePassingRules._prepare(rng, 2, layerf)

        @test typeof(outt[1]) <: FlowMessagePassingRules.AdditiveCouplingLayerEmpty
        @test typeof(outt[2]) <: FlowMessagePassingRules.PermutationLayer
        @test typeof(outf) <: FlowMessagePassingRules.AdditiveCouplingLayerEmpty

        @test outt[1].dim == 2
        @test outt[2].dim == 2
        @test outf.dim == 2

        layer_comp = compile(rng, outf)
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

        @test nr_params(outf) == 3
        @test nr_params(layer_comp) == 3

        # Several coupling flows: one per partition after the first, each of its own parameters.
        scalar = FlowMessagePassingRules._prepare(rng, 4, AdditiveCouplingLayer(PlanarFlow(); permute = false))
        @test length(scalar.f) == 3 && nr_params(scalar) == 3 * 3 && nr_params(compile(rng, scalar)) == 9
        blocks = FlowMessagePassingRules._prepare(rng, 6, AdditiveCouplingLayer(PlanarFlow(); partition_dim = 2, permute = false))
        @test length(blocks.f) == 2 && nr_params(blocks) == 2 * (2 * 2 + 1)
        # The partitions must tile the input.
        @test_throws "needs a model dimension that is a multiple of it; got 5" FlowMessagePassingRules._prepare(rng, 5, AdditiveCouplingLayer(PlanarFlow(); partition_dim = 2))
    end

    @testset "Get" begin

        # check get functions for univariate PlanarFlow
        f = PlanarFlow()
        layer = AdditiveCouplingLayer(f; permute = false)
        out = compile(rng, FlowMessagePassingRules._prepare(rng, 2, layer))
        @test getf(out) == out.f
        @test getflow(out) == out.f
        @test getdim(out) == out.dim
        @test getdim(out) == 2
    end

    @testset "Base" begin

        # check base functions (univariate)
        f = PlanarFlow()
        layer = AdditiveCouplingLayer(f; permute = false)
        out = compile(rng, FlowMessagePassingRules._prepare(rng, 2, layer))
        @test eltype(out) == Float64
    end

    @testset "Forward-Backward" begin

        # check forward function
        params = [1.0, 2.0, -3.0]
        f = PlanarFlow()
        layer = AdditiveCouplingLayer(f; permute = false)
        layer = compile(FlowMessagePassingRules._prepare(rng, 2, layer), params)
        @test forward(layer, [5.0, 1.5]) == [5.0, 7.4999983369439445]
        @test forward(layer, [4.0, 2.5]) == [4.0, 7.499909204262595]
        @test forward.(layer, [[5.0, 1.5], [4.0, 2.5]]) ==
            [[5.0, 7.4999983369439445], [4.0, 7.499909204262595]]

        # check forward! function
        params = [1.0, 2.0, -3.0]
        f = PlanarFlow()
        layer = AdditiveCouplingLayer(f; permute = false)
        layer = compile(FlowMessagePassingRules._prepare(rng, 2, layer), params)
        output = zeros(2)
        forward!(output, layer, [5.0, 1.5])
        @test output == [5.0, 7.4999983369439445]
        forward!(output, layer, [4.0, 2.5])
        @test output == [4.0, 7.499909204262595]

        # check forward function (input > 2)
        f = PlanarFlow()
        layer = AdditiveCouplingLayer(f; permute = false)
        layer = compile(rng, FlowMessagePassingRules._prepare(rng, 3, layer))
        x = randn(rng, 3)
        @test backward(layer, forward(layer, x)) ≈ x

        # check backward function
        params = [1.0, 2.0, -3.0]
        f = PlanarFlow()
        layer = AdditiveCouplingLayer(f; permute = false)
        layer = compile(FlowMessagePassingRules._prepare(rng, 2, layer), params)
        @test backward(layer, [5.0, 7.4999983369439445]) == [5.0, 1.5]
        @test backward(layer, [4.0, 7.499909204262595]) == [4.0, 2.5]
        @test backward.(
            layer, [[5.0, 7.4999983369439445], [4.0, 7.499909204262595]]
        ) == [[5.0, 1.5], [4.0, 2.5]]

        # check backward! function
        params = [1.0, 2.0, -3.0]
        f = PlanarFlow()
        layer = AdditiveCouplingLayer(f; permute = false)
        layer = compile(FlowMessagePassingRules._prepare(rng, 2, layer), params)
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
        layer = compile(FlowMessagePassingRules._prepare(rng, 2, layer), params)
        @test jacobian(layer, [3.0, 1.5]) == [1.0 0.0; 1.0197320743308804 1.0]
        @test jacobian(layer, [2.5, 5.0]) == [1.0 0.0; 1.1413016497063289 1.0]
        @test jacobian.(layer, [[3.0, 1.5], [2.5, 5.0]]) == [
            [1.0 0.0; 1.0197320743308804 1.0], [1.0 0.0; 1.1413016497063289 1.0],
        ]

        # check jacobian function
        params = [1.0, 2.0, -3.0]
        f = PlanarFlow()
        layer = AdditiveCouplingLayer(f; permute = false)
        layer = compile(FlowMessagePassingRules._prepare(rng, 2, layer), params)
        @test inv_jacobian(layer, [3.0, 1.5]) ==
            [1.0 0.0; -1.0197320743308804 1.0]
        @test inv_jacobian(layer, [2.5, 5.0]) ==
            [1.0 0.0; -1.1413016497063289 1.0]
        @test inv_jacobian.(layer, [[3.0, 1.5], [2.5, 5.0]]) == [
            [1.0 0.0; -1.0197320743308804 1.0],
            [1.0 0.0; -1.1413016497063289 1.0],
        ]

        # check for invertibility
        layer = AdditiveCouplingLayer(PlanarFlow(); permute = false)
        x = randn(rng, 10)
        layer = compile(rng, FlowMessagePassingRules._prepare(rng, 10, layer))
        @test inv(jacobian(layer, x)) ≈ inv_jacobian(layer, forward(layer, x))
    end

    @testset "Utility Jacobian" begin

        # check utility functions jacobian (univariate)
        params = [1.0, 2.0, -3.0]
        f = PlanarFlow()
        layer = AdditiveCouplingLayer(f; permute = false)
        layer = compile(FlowMessagePassingRules._prepare(rng, 2, layer), params)
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

@testitem "models:Additive Coupling Layer over partitions of several coordinates" tags = [:models] begin
    using FlowMessagePassingRules, ForwardDiff, LinearAlgebra, StableRNGs
    using FlowMessagePassingRules: forward, backward, jacobian, inv_jacobian, logabsdet_jacobian

    # y₁ = x₁, yₖ = xₖ + fₖ₋₁(xₖ₋₁) over partitions of `pdim` coordinates: every pass agrees with
    # the map written out, and the Jacobians with ForwardDiff's of `forward` and `backward`.
    rng = StableRNG(7)
    for flow in (PlanarFlow(), RadialFlow()), (dim, pdim) in ((2, 1), (4, 1), (4, 2), (6, 2), (6, 3), (6, 6))
        @testset "$(flow isa FlowMessagePassingRules.PlanarFlowPlaceholder ? "planar" : "radial"), dim = $dim, partition_dim = $pdim" begin
            model = compile(rng, FlowModel(dim, (AdditiveCouplingLayer(flow; partition_dim = pdim, permute = false),)))
            layer = only(model.layers)
            blocks = dim ÷ pdim
            @test length(layer.f) == blocks - 1
            block(k) = (1 + (k - 1) * pdim):(k * pdim)
            x = randn(rng, dim)
            expected = copy(x)
            for k in 2:blocks
                expected[block(k)] .+= forward(layer.f[k - 1], pdim == 1 ? x[block(k - 1)][1] : x[block(k - 1)])
            end
            y = forward(layer, x)
            @test y ≈ expected
            @test backward(layer, y) ≈ x
            @test Matrix(jacobian(layer, x)) ≈ ForwardDiff.jacobian(z -> forward(layer, z), x)
            @test Matrix(inv_jacobian(layer, y)) ≈ ForwardDiff.jacobian(z -> backward(layer, z), y)
            @test Matrix(inv_jacobian(layer, y)) * Matrix(jacobian(layer, x)) ≈ I
            @test logabsdet_jacobian(layer, x) == 0
            @test first(logabsdet(Matrix(jacobian(layer, x)))) ≈ 0 atol = 1.0e-12
        end
    end

    # A coupling flow over a block takes any vector, a view or a dual number's.
    f = PlanarFlow(StableRNG(3), 3)
    x = randn(StableRNG(4), 5)
    @test forward(f, view(x, 2:4)) ≈ forward(f, x[2:4])
    @test ForwardDiff.jacobian(z -> forward(f, z), x[2:4]) ≈ jacobian(f, x[2:4])
    r = RadialFlow(StableRNG(5), 3)
    @test forward(r, view(x, 2:4)) ≈ forward(r, x[2:4])
    @test ForwardDiff.jacobian(z -> forward(r, z), x[2:4]) ≈ jacobian(r, x[2:4])
end
