# Engine trajectories

An engine is tested by what a whole inference run produces, its trajectory. A trajectory holds
the [Bethe free energy](@extref MessagePassingRulesBase glossary-bethe-free-energy) after each
iteration, the final posteriors, and every rule call in the order it happened, with its result
and [log scale](@extref MessagePassingRulesBase glossary-log-scale). Comparing trajectories
checks the schedule that reached the answer as well as the answer, since the order in which
messages are emitted is part of what an engine must reproduce.

A run is recorded as an [`EngineTrajectory`](@ref) of [`RuleCallRecord`](@ref)s. An engine
typically records them from its rule-call callbacks. [`save_engine_fixture`](@ref) saves a
trajectory, [`load_engine_fixture`](@ref) reads it back, and
[`compare_engine_trajectory`](@ref) compares it with a new run.

## Record a run

This page needs no engine. A small function plays one: it infers the mean `x` of the `Gaussian`
node of the [overview](@ref "A node to test") from two observations, with a normal prior on
`x`. Each iteration calls the rule towards `μ` once per observation, records the call, and
multiplies the messages into the posterior.

```@example engine
using MessagePassingRulesBase, MessagePassingRulesTestUtils
using BayesBase, ExponentialFamily, Distributions, Test

struct Gaussian end
Gaussian(μ, v) = NormalMeanVariance(μ, v)

@define_factor_node(node = Gaussian, type = Stochastic, interfaces = [:out, :μ, :v])

@define_message_update_rule(
    node = Gaussian, target = :μ,
    args = (m[:out]::PointMass, m[:v]::PointMass),
    logscale = 0,
    body = (args) -> NormalMeanVariance(mean(args.m[:out]), mean(args.m[:v])),
)

function toy_run(id, ys; prior = NormalMeanVariance(0.0, 10.0), v = 0.5, iterations = 2)
    trace = RuleCallRecord[]
    free_energy = Float64[]
    posterior = prior
    for iteration in 1:iterations
        posterior = prior
        for y in ys
            message = @call_message_update_rule(node = Gaussian, target = :μ, m = (out = PointMass(y), v = PointMass(v)))
            push!(trace, RuleCallRecord(iteration, "Gaussian", "μ", getresult(message), getlogscale(message)))
            posterior = prod(PreserveTypeProd(Distribution), posterior, getresult(message))
        end
        # For this exact model, the free energy is the negative log evidence.
        push!(free_energy, -logpdf(MvNormal(fill(mean(prior), 2), var(prior) .+ [v 0; 0 v]), ys))
    end
    return EngineTrajectory(id; free_energy, posteriors = Dict("x" => posterior), trace)
end

reference = toy_run("toy", [1.0, 2.0])
reference.trace
```

## Save and load a fixture

```@example engine
path = save_engine_fixture(tempname() * ".toml", reference; packages = Dict("MyEngine" => v"1.0.0"))
print(read(path, String))
```

The file is TOML, and every value in it is in the portable form that
[`encode_fixture_value`](@ref) gives: family names and parameters, no Julia types. Any Julia
version, and any release of the engine, reads a fixture that another recorded.

```@example engine
header, recorded = load_engine_fixture(path)
header
```

## Compare a run

A new run is compared with the recorded one. The free energy, each posterior and the trace are
each a `Test` assertion:

```@example engine
@testset "toy trajectory" begin
    @test compare_engine_trajectory(toy_run("toy", [1.0, 2.0]), recorded) === :agree
end
nothing # hide
```

A new engine may deliberately reorder independent calls. Here the run processes the
observations in the other order, so each iteration makes the same calls in another order.
`trace_order = :within_iteration` compares each iteration's calls in any order:

```@example engine
reordered = toy_run("toy", [2.0, 1.0])
compare_engine_trajectory(reordered, recorded; trace_order = :within_iteration)
```

The reference engine may compute a message once per subscriber, where the new engine shares it.
`collapse_repeats = true` drops the repeats from the reference trace:

```@example engine
repeated = EngineTrajectory(
    "toy"; free_energy = recorded.free_energy, posteriors = recorded.posteriors,
    trace = [call for call in recorded.trace for _ in 1:2],
)
compare_engine_trajectory(toy_run("toy", [1.0, 2.0]), repeated; collapse_repeats = true)
```

A [`DeclaredDisagreement`](@ref) with the reference's `id` turns a difference into a recorded
outcome, as it does for [`compare_with_reference`](@ref).

## API

```@docs
EngineTrajectory
RuleCallRecord
compare_engine_trajectory
save_engine_fixture
load_engine_fixture
encode_fixture_value
```
