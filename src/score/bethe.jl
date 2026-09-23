export bethe_free_energy

"""
    bethe_free_energy(::Type{T}, factornodes, variables; algorithm = (node) -> nothing)

The stream of the Bethe free energy of a graph, as values of type `T`: the sum of every
factor node's [`FactorBoundFreeEnergy`](@ref), plus the [`VariableBoundEntropy`](@ref) of
every random variable, minus one point entropy per connection of a data or constant variable.

`variables` may hold variables of every kind; each is counted by what it is. `algorithm(node)`
is the algorithm the node runs under, as given at activation; `nothing` is the node's default.

The stream emits when every component has a value and again whenever one of them updates,
which is once per iteration when the data is fed once per iteration.
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
