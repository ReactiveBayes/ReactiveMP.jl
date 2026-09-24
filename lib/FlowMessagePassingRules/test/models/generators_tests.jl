# What the port adds: an explicit generator for everything that draws, the model's random
# permutations and its initial parameters, defaulting to the task's.

@testitem "models:generators, compile is reproducible" tags = [:models] begin
    using FlowMessagePassingRules, StableRNGs
    using FlowMessagePassingRules: forward, jacobian

    layers(dim) = (
        InputLayer(dim),
        AdditiveCouplingLayer(PlanarFlow()),
        AdditiveCouplingLayer(RadialFlow(); permute = false),
        PermutationLayer(),
        AdditiveCouplingLayer(PlanarFlow(); permute = false),
    )

    for dim in (2, 3, 5)
        # The same generator gives the same model, permutations and parameters.
        model = FlowModel(StableRNG(dim), layers(dim))
        a, b = compile(StableRNG(1), model), compile(StableRNG(1), model)
        @test typeof(a) == typeof(b)
        x = randn(StableRNG(2), dim)
        @test forward(a, x) == forward(b, x)
        @test jacobian(a, x) == jacobian(b, x)
        # and so does rebuilding the model from the same generator
        other = FlowModel(StableRNG(dim), layers(dim))
        @test map(l -> l isa PermutationLayer ? l.P.ind : nothing, getlayers(model)) ==
            map(l -> l isa PermutationLayer ? l.P.ind : nothing, getlayers(other))
        @test forward(compile(StableRNG(1), other), x) == forward(a, x)

        # A different one gives different parameters.
        @test forward(compile(StableRNG(3), model), x) != forward(a, x)

        # Without a generator, the task's is used.
        @test compile(model) isa CompiledFlowModel
        @test compile(FlowModel(layers(dim))) isa CompiledFlowModel
        @test nr_params(compile(model)) == nr_params(model)
    end

    # FlowModel(rng, dim, layers) is the same as with an InputLayer.
    placeholders = (AdditiveCouplingLayer(PlanarFlow()), AdditiveCouplingLayer(PlanarFlow()))
    m1 = compile(StableRNG(4), FlowModel(StableRNG(5), 4, placeholders))
    m2 = compile(StableRNG(4), FlowModel(StableRNG(5), (InputLayer(4), placeholders...)))
    @test forward(m1, [1.0, 2.0, 3.0, 4.0]) == forward(m2, [1.0, 2.0, 3.0, 4.0])
    @test typeof(FlowModel(4, placeholders)) == typeof(FlowModel(StableRNG(5), 4, placeholders))

    # Explicit parameters draw nothing from any generator: the permutations are the model's.
    model = FlowModel(StableRNG(6), 4, placeholders)
    params = collect(1.0:nr_params(model)) ./ 10
    @test forward(compile(model, params), ones(4)) == forward(compile(model, params), ones(4))
end

@testitem "models:generators, layers and coupling flows" tags = [:models] begin
    using FlowMessagePassingRules, StableRNGs
    using FlowMessagePassingRules: PlanarFlowEmpty, RadialFlowEmpty, getall, getP

    # The permutation layer's random matrix.
    @test getP(PermutationLayer(StableRNG(1), 6)) == getP(PermutationLayer(StableRNG(1), 6))
    @test getP(PermutationLayer(StableRNG(1), 6)) == PermutationMatrix(StableRNG(1), 6)
    @test PermutationLayer(6) isa PermutationLayer

    # The coupling flows, drawn directly and compiled from their empty forms.
    for dim in (1, 2, 5)
        @test getall(PlanarFlow(StableRNG(dim), dim)) == getall(PlanarFlow(StableRNG(dim), dim))
        @test getall(compile(StableRNG(dim), PlanarFlowEmpty(dim))) ==
            getall(compile(StableRNG(dim), PlanarFlowEmpty(dim)))
        @test getall(compile(StableRNG(dim), RadialFlowEmpty(dim))) ==
            getall(compile(StableRNG(dim), RadialFlowEmpty(dim)))
        @test PlanarFlow(dim) isa PlanarFlow
        @test compile(PlanarFlowEmpty(dim)) isa PlanarFlow
        @test compile(RadialFlowEmpty(dim)) isa RadialFlow
    end
    for dim in (2, 5)
        @test getall(RadialFlow(StableRNG(dim), dim)) == getall(RadialFlow(StableRNG(dim), dim))
        @test RadialFlow(dim) isa RadialFlow
    end
    # the same draws as v6's, in the same order, from the generator given
    rng = StableRNG(7)
    u, w, b = randn(rng, 3), randn(rng, 3), randn(rng)
    @test getall(PlanarFlow(StableRNG(7), 3)) == (u, w, b)
    rng = StableRNG(7)
    z0, α, β = randn(rng, 3), rand(rng), randn(rng)
    @test getall(RadialFlow(StableRNG(7), 3)) == (z0, α, β)
end

@testitem "models:a model without layers" tags = [:models] begin
    using FlowMessagePassingRules

    # An empty tuple is both a tuple of layers and a tuple of placeholders, which v6 found
    # ambiguous.
    model = FlowModel(3, ())
    @test getlayers(model) == ()
    @test nr_params(model) == 0
end
