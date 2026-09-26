export bethe_free_energy

"""
    bethe_free_energy(::Type{T}, factornodes, variables; algorithm = (node) -> nothing) where {T <: Real}

The stream of the Bethe free energy of an activated graph, as values of type `T`:

```math
F[q] = \\sum_a \\left( \\mathbb{E}_{q_a}[-\\log f_a] - H[q_a] \\right) + \\sum_x (d_x - 1) H[q_x],
```

the sum of every factor node's [`FactorBoundFreeEnergy`](@ref), plus the
[`VariableBoundEntropy`](@ref) of every random variable, minus one point entropy per connection of
a data or constant variable, which cancels the point masses' infinite entropies in the nodes'
terms.

# Arguments

- `factornodes`: every factor node of the graph;
- `variables`: every variable of the graph, the data and the constants included: each is
  counted by what it is, and a data or constant variable left out leaves its point entropies
  uncancelled, which makes the free energy infinite.

# Keywords

- `algorithm`: a function of a node, the algorithm the node runs under, as given at activation.
  Default `(node) -> nothing`, every node's default.

# Returns

A Rocket.jl observable of `T`. It emits once every component has a value, initial marginals
skipped, and again whenever one of them updates: once per iteration when the data is given once
per iteration. Subscribe to it before giving the data.

# Throws

The stream fails when it computes a value for a node with no average energy for its clusters'
marginals, with a [`RuleNotFoundError`](@extref MessagePassingRulesBase.RuleNotFoundError) naming
the node. The average energies run with the engine's context,
[`ReactiveMP.node_context`](@ref)`(node)`: the services of the activation option `context` do not
reach them, and one that declares a service the engine does not supply fails.

# Examples

```julia
energies = Float64[]
subscription = subscribe!(bethe_free_energy(Float64, nodes, variables), (f) -> push!(energies, f))
```
"""
function bethe_free_energy(::Type{T}, factornodes, variables; algorithm = (node) -> nothing) where {T <: Real}
    CT = counting_real_type(T)

    node_bound_free_energies = map(node -> score(CT, FactorBoundFreeEnergy(), node, algorithm(node), nothing), collect(factornodes))
    randomvars = filter(israndom, collect(variables))
    variable_bound_entropies = map(variable -> score(CT, VariableBoundEntropy(), variable, nothing), randomvars)

    sum_of_latest = (values) -> reduce(+, values)
    node_bound_free_energies_sum = collectLatest(CT, CT, node_bound_free_energies, sum_of_latest)
    variable_bound_entropies_sum = collectLatest(CT, CT, variable_bound_entropies, sum_of_latest)

    point_entropies_n = mapreduce(degree, +, Iterators.filter(v -> isdata(v) || isconst(v), variables); init = 0)
    point_entropies = CountingReal(T, point_entropies_n)

    return combineLatest((node_bound_free_energies_sum, variable_bound_entropies_sum), PushNew()) |>
        map(T, sums -> float(sums[1] + sums[2] - point_entropies))
end

counting_real_type(::Type{Real}) = CountingReal{<:Real}
counting_real_type(::Type{T}) where {T <: Real} = CountingReal{T}
