# Engine trajectories

An engine is tested by what a whole inference run produces, its trajectory: the free energy per
iteration, the final posteriors, and every rule call in the order it happened, with its result
and log scale. Comparing trajectories checks not only the answer but the schedule that reached
it, since the order in which messages are emitted is part of what an engine must reproduce.

A run is recorded as an [`EngineTrajectory`](@ref) of [`RuleCallRecord`](@ref)s, typically from
the engine's rule-call callbacks, saved with [`save_engine_fixture`](@ref), and later read with
[`load_engine_fixture`](@ref) and compared with a new run by
[`compare_engine_trajectory`](@ref):

```julia
trajectory = EngineTrajectory("coin-toss"; free_energy, posteriors = (θ = q_θ,), trace)
save_engine_fixture("coin_toss.toml", trajectory; packages = Dict("ReactiveMP" => v"6.5.0"))

header, reference = load_engine_fixture("coin_toss.toml")
compare_engine_trajectory(new_trajectory, reference)
```

The file is TOML, with every value in the portable form [`encode_fixture_value`](@ref) gives:
family names and parameters, no Julia types. A fixture recorded on one Julia version, or by an
older release of the engine, is therefore read by any other. Where a new engine deliberately
reorders independent calls, `trace_order = :within_iteration` compares each iteration's calls
in any order; where the reference computes a message once per subscriber and the new engine
shares it, `collapse_repeats = true` drops the repeats.

## API

```@docs
EngineTrajectory
RuleCallRecord
compare_engine_trajectory
save_engine_fixture
load_engine_fixture
encode_fixture_value
```
