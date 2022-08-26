using Unrolled

"""
    CombinedRange{L, R}

`CombinedRange` represents a range of combined variable in factorisation specification language. Such variables specified to be in the same factorisation cluster.

See also: [`ReactiveMP.SplittedRange`](@ref)
"""
struct CombinedRange{L, R}
    from :: L
    to   :: R
end

Base.firstindex(range::CombinedRange) = range.from
Base.lastindex(range::CombinedRange)  = range.to
Base.in(item, range::CombinedRange)   = firstindex(range) <= item <= lastindex(range)

Base.show(io::IO, range::CombinedRange) = print(io, repr(range.from), ":", repr(range.to))

## 

"""
    SplittedRange{L, R}

`SplittedRange` represents a range of splitted variable in factorisation specification language. Such variables specified to be **not** in the same factorisation cluster.

See also: [`ReactiveMP.CombinedRange`](@ref)
"""
struct SplittedRange{L, R}
    from :: L
    to   :: R
end

is_splitted(any)                  = false
is_splitted(range::SplittedRange) = true

Base.firstindex(range::SplittedRange) = range.from
Base.lastindex(range::SplittedRange)  = range.to
Base.in(item, range::SplittedRange)   = firstindex(range) <= item <= lastindex(range)

Base.show(io::IO, range::SplittedRange) = print(io, repr(range.from), "..", repr(range.to))

## 

"""
    __as_unit_range

Converts a value to a `UnitRange`. This function is a part of private API and is not intended for public usage.
"""
function __as_unit_range end

__as_unit_range(any)                  = error("Internal error: Cannot represent $(any) as unit range.")
__as_unit_range(index::Integer)       = index:index
__as_unit_range(range::CombinedRange) = firstindex(range):lastindex(range)
__as_unit_range(range::SplittedRange) = firstindex(range):lastindex(range)

## 

"""
    FactorisationSpecificationNotDefinedYet{S}

Internal structure to denote not defined factorisation specification in constraints specification language. 
**Note**: this structure is a part of private API and is not intended for public use.
"""
struct FactorisationSpecificationNotDefinedYet{S} end

## 

struct FactorisationConstraintsEntry{N, I} end

FactorisationConstraintsEntry(::Val{N}, ::Val{I}) where {N, I} = FactorisationConstraintsEntry{N, I}()

getnames(entry::FactorisationConstraintsEntry{N}) where {N}         = N
getindices(entry::FactorisationConstraintsEntry{N, I}) where {N, I} = I
getpairs(entry::FactorisationConstraintsEntry)                      = zip(getnames(entry), getindices(entry))

__io_entry_pair(pair::Tuple)                            = __io_entry_pair(pair[1], pair[2])
__io_entry_pair(symbol::Symbol, ::Nothing)              = string(symbol)
__io_entry_pair(symbol::Symbol, index::Integer)         = string(symbol, "[", index, "]")
__io_entry_pair(symbol::Symbol, index::FunctionalIndex) = string(symbol, "[", repr(index), "]")
__io_entry_pair(symbol::Symbol, range::CombinedRange)   = string(symbol, "[", range, "]")
__io_entry_pair(symbol::Symbol, range::SplittedRange)   = string(symbol, "[", range, "]")

function Base.show(io::IO, entry::FactorisationConstraintsEntry)
    print(io, "q(")
    join(io, Iterators.map(__io_entry_pair, getpairs(entry)), ", ")
    print(io, ")")
end

## 

struct FactorisationConstraintsSpecification{N, E} end

FactorisationConstraintsSpecification(::Val{N}, ::Val{E}) where {N, E}    = FactorisationConstraintsSpecification{N, E}()
FactorisationConstraintsSpecification(::Val{N}, ::Val{nothing}) where {N} = error("Cannot create q(", join(N, ","), ") factorisation constraints specification")

getnames(specification::FactorisationConstraintsSpecification{N}) where {N}         = N
getentries(specification::FactorisationConstraintsSpecification{N, E}) where {N, E} = E

Base.:(*)(
    left::Tuple{Vararg{T where T <: Union{<:FactorisationConstraintsSpecification, <:FactorisationConstraintsEntry}}},
    right::Tuple{Vararg{T where T <: Union{<:FactorisationConstraintsSpecification, <:FactorisationConstraintsEntry}}}
) = (left..., right...)
Base.:(*)(
    left::Union{<:FactorisationConstraintsSpecification, <:FactorisationConstraintsEntry},
    right::Tuple{Vararg{T where T <: Union{<:FactorisationConstraintsSpecification, <:FactorisationConstraintsEntry}}}
) = (left, right...)
Base.:(*)(
    left::Tuple{Vararg{T where T <: Union{<:FactorisationConstraintsSpecification, <:FactorisationConstraintsEntry}}},
    right::Union{<:FactorisationConstraintsSpecification, <:FactorisationConstraintsEntry}
) = (left..., right)
Base.:(*)(
    left::Union{<:FactorisationConstraintsSpecification, <:FactorisationConstraintsEntry},
    right::Union{<:FactorisationConstraintsSpecification, <:FactorisationConstraintsEntry}
) = (left, right)

Base.:(*)(::FactorisationSpecificationNotDefinedYet{S}, something::Any) where {S}                                      = error("Cannot multiply $S and $something. $S has not been defined yet.")
Base.:(*)(something::Any, ::FactorisationSpecificationNotDefinedYet{S}) where {S}                                      = error("Cannot multiply $S and $something. $S has not been defined yet.")
Base.:(*)(::FactorisationSpecificationNotDefinedYet{S1}, ::FactorisationSpecificationNotDefinedYet{S2}) where {S1, S2} = error("Cannot multiply $S1 and $S2. Both $S1 and $S2 have not been defined yet.")

function Base.show(io::IO, factorisation::FactorisationConstraintsSpecification{Names}) where {Names}
    print(io, "q(")
    join(io, getnames(factorisation), ", ")
    print(io, ")")

    iscompact = get(io, :compact, false)

    if !iscompact
        print(io, " = ")
        foreach(getentries(factorisation)) do e
            print(IOContext(io, :compact => true), e)
        end
    end
end

# Split related functions

Base.:(*)(
    left::Tuple{Vararg{T where T <: FactorisationConstraintsEntry}},
    right::Tuple{Vararg{T where T <: FactorisationConstraintsEntry}}
) = (left..., right...)

# Only these combinations are allowed to be merged
__factorisation_split_merge_range(a::Int, b::Int)                         = SplittedRange(a, b)
__factorisation_split_merge_range(a::FunctionalIndex, b::Int)             = SplittedRange(a, b)
__factorisation_split_merge_range(a::Int, b::FunctionalIndex)             = SplittedRange(a, b)
__factorisation_split_merge_range(a::FunctionalIndex, b::FunctionalIndex) = SplittedRange(a, b)
__factorisation_split_merge_range(a::Any, b::Any)                         = error("Cannot merge $(a) and $(b) indexes in `factorisation_split`")

function factorisation_split(
    left::Tuple{Vararg{T where T <: FactorisationConstraintsEntry}},
    right::Tuple{Vararg{T where T <: FactorisationConstraintsEntry}}
)
    left_last   = last(left)
    right_first = first(right)
    (getnames(left_last) === getnames(right_first)) ||
        error("Cannot split $(left_last) and $(right_first). Names or their order does not match.")
    (length(getnames(left_last)) === length(Set(getnames(left_last)))) ||
        error("Cannot split $(left_last) and $(right_first). Names should be unique.")
    lindices = getindices(left_last)
    rindices = getindices(right_first)
    split_merged = unrolled_map(__factorisation_split_merge_range, lindices, rindices)

    # This check happens at runtime
    # first_split = first(split_merged)
    # unrolled_all(e -> e === first_split, split_merged) || error("Inconsistent indices within factorisation split. Check $(split_merged) indices for $(getnames(left_last)) variables.")

    return (
        left[1:end-1]...,
        FactorisationConstraintsEntry(Val(getnames(left_last)), Val(split_merged)),
        right[begin+1:end]...
    )
end

## 

"""
    __factorisation_specification_resolve_index(index, collection)

This function materializes index from constraints specification to something we can use `Base.in` function to. For example constraint specification index may return `begin` or `end`
placeholders in a form of the `FunctionalIndex` structure. This function correctly resolves all indices and check bounds as an extra step.
"""
function __factorisation_specification_resolve_index end

__factorisation_specification_resolve_index(index::Any, collection::AbstractVariable)                              = error("Attempt to access a single variable $(name(collection)) at index [$(index)].") # `index` here is guaranteed to be not `nothing`, because of dispatch. `Nothing, Nothing` version will dispatch on the method below
__factorisation_specification_resolve_index(index::Nothing, collection::AbstractVariable)                          = nothing
__factorisation_specification_resolve_index(index::Nothing, collection::AbstractArray{<:AbstractVariable})         = nothing
__factorisation_specification_resolve_index(index::Real, collection::AbstractArray{<:AbstractVariable})            = error("Non integer indices are not supported. Attempt to access collection $(collection) of variable $(name(first(collection))) at index [$(index)].")
__factorisation_specification_resolve_index(index::Integer, collection::AbstractArray{<:AbstractVariable})         = (firstindex(collection) <= index <= lastindex(collection)) ? index : error("Index out of bounds happened during indices resolution in factorisation constraints. Attempt to access collection $(collection) of variable $(name(first(collection))) at index [$(index)].")
__factorisation_specification_resolve_index(index::FunctionalIndex, collection::AbstractArray{<:AbstractVariable}) = __factorisation_specification_resolve_index(index(collection)::Integer, collection)::Integer
__factorisation_specification_resolve_index(index::CombinedRange, collection::AbstractArray{<:AbstractVariable})   = CombinedRange(__factorisation_specification_resolve_index(firstindex(index), collection)::Integer, __factorisation_specification_resolve_index(lastindex(index), collection)::Integer)
__factorisation_specification_resolve_index(index::SplittedRange, collection::AbstractArray{<:AbstractVariable})   = SplittedRange(__factorisation_specification_resolve_index(firstindex(index), collection)::Integer, __factorisation_specification_resolve_index(lastindex(index), collection)::Integer)

## Some pre-written optimised dispatch rules for the `UnspecifiedConstraints` case

resolve_factorisation(::UnspecifiedConstraints, allvariables, fform, variables)      = resolve_factorisation(UnspecifiedConstraints(), sdtype(fform), allvariables, fform, variables)
resolve_factorisation(::UnspecifiedConstraints, any, allvariables, fform, variables) = resolve_factorisation(__EmptyConstraints, allvariables, fform, variables)

# Preoptimised dispatch rule for unspecified constraints and a deterministic node with any number of inputs
resolve_factorisation(::UnspecifiedConstraints, ::Deterministic, allvariables, fform, variables) = FullFactorisation()

# Preoptimised dispatch rules for unspecified constraints and a stochastic node with 2 inputs
resolve_factorisation(::UnspecifiedConstraints, ::Stochastic, allvariables, fform, ::Tuple{V1, V2}) where {V1 <: RandomVariable, V2 <: RandomVariable}                         = ((1, 2),)
resolve_factorisation(::UnspecifiedConstraints, ::Stochastic, allvariables, fform, ::Tuple{V1, V2}) where {V1 <: Union{<:ConstVariable, <:DataVariable}, V2 <: RandomVariable} = ((1,), (2,))
resolve_factorisation(::UnspecifiedConstraints, ::Stochastic, allvariables, fform, ::Tuple{V1, V2}) where {V1 <: RandomVariable, V2 <: Union{<:ConstVariable, <:DataVariable}} = ((1,), (2,))

# Preoptimised dispatch rules for unspecified constraints and a stochastic node with 3 inputs
resolve_factorisation(::UnspecifiedConstraints, ::Stochastic, allvariables, fform, ::Tuple{V1, V2, V3}) where {V1 <: RandomVariable, V2 <: RandomVariable, V3 <: RandomVariable}                                                 = ((1, 2, 3),)
resolve_factorisation(::UnspecifiedConstraints, ::Stochastic, allvariables, fform, ::Tuple{V1, V2, V3}) where {V1 <: Union{<:ConstVariable, <:DataVariable}, V2 <: RandomVariable, V3 <: RandomVariable}                         = ((1,), (2, 3))
resolve_factorisation(::UnspecifiedConstraints, ::Stochastic, allvariables, fform, ::Tuple{V1, V2, V3}) where {V1 <: RandomVariable, V2 <: Union{<:ConstVariable, <:DataVariable}, V3 <: RandomVariable}                         = ((1, 3), (2,))
resolve_factorisation(::UnspecifiedConstraints, ::Stochastic, allvariables, fform, ::Tuple{V1, V2, V3}) where {V1 <: RandomVariable, V2 <: RandomVariable, V3 <: Union{<:ConstVariable, <:DataVariable}}                         = ((1, 2), (3,))
resolve_factorisation(::UnspecifiedConstraints, ::Stochastic, allvariables, fform, ::Tuple{V1, V2, V3}) where {V1 <: RandomVariable, V2 <: Union{<:ConstVariable, <:DataVariable}, V3 <: Union{<:ConstVariable, <:DataVariable}} = ((1,), (2,), (3,))
resolve_factorisation(::UnspecifiedConstraints, ::Stochastic, allvariables, fform, ::Tuple{V1, V2, V3}) where {V1 <: Union{<:ConstVariable, <:DataVariable}, V2 <: RandomVariable, V3 <: Union{<:ConstVariable, <:DataVariable}} = ((1,), (2,), (3,))
resolve_factorisation(::UnspecifiedConstraints, ::Stochastic, allvariables, fform, ::Tuple{V1, V2, V3}) where {V1 <: Union{<:ConstVariable, <:DataVariable}, V2 <: Union{<:ConstVariable, <:DataVariable}, V3 <: RandomVariable} = ((1,), (2,), (3,))

"""
    resolve_factorisation(constraints, allvariables, fform, variables) 

This function resolves factorisation constraints in a form of a tuple for a given `constraints`, `allvariables`, `fform`, and `variables`.

See also: [`ConstraintsSpecification`](@ref)
"""
function resolve_factorisation end

function resolve_factorisation(constraints, allvariables, fform, variables)
    return resolve_factorisation(sdtype(fform), constraints, allvariables, fform, variables)
end

# Deterministic nodes always have `FullFactorisation` constraint (by default)
function resolve_factorisation(::Deterministic, constraints, allvariables, fform, variables)
    return FullFactorisation()
end

# Stochastic nodes may have different factorisation constraints
function resolve_factorisation(::Stochastic, constraints, allvariables, fform, _variables)
    # Input `_variables` may include 'tupled' variables in it (e.g. in NormalMixture node)
    # Before doing any computations we flatten the input and perform all computations in flatten space
    # The output of the `resolve_factorisation` is flattened too
    # TODO: This approach does not really work for "array"-ed variables, but we do not support this currently anyway
    variables = TupleTools.flatten(_variables)
    N         = length(variables)

    preallocated = constraints.preallocated

    __reset_preallocated!(preallocated, N)

    clusters_template  = preallocated.clusters_template
    clusters_usage     = preallocated.clusters_usage
    clusters_set       = preallocated.clusters_set
    cluster_indices    = preallocated.cluster_indices
    var_refs_positions = preallocated.var_refs_positions

    var_refs = map(resolve_variable_proxy, variables)

    var_refs_names   = map(r -> r[1], var_refs)
    var_refs_indices = map(r -> r[2], var_refs)

    vardict = getvardict(allvariables)

    var_refs_collections = map(var_refs_names) do name
        return get(
            () -> error(
                "Variables collection has no variable named $(name). Double check the expression `$(var_refs_names[1]) ~ $(fform)($(join(var_refs_names[2:end], ", ")))`."
            ),
            vardict,
            name
        )
    end

    # function _resolve_var_ref_position
    #     return is_found, index, range, is_splitted
    # end

    # Note from bvdmitri: see explanation about explicit `::Tuple{Bool, Int, UnitRange{Int}, Bool}` further in main filter procedure
    function __resolve_var_ref_position(
        qpair_name::Symbol,
        qpair_index::Nothing,
        start_with::Int
    )::Tuple{Bool, Int, UnitRange{Int}, Bool}
        position    = findnext(==(qpair_name), var_refs_names, start_with)
        is_found    = position !== nothing
        _position   = is_found ? position : 0
        range       = typemin(Int):typemax(Int)
        is_splitted = false
        return (is_found, _position, range, is_splitted)::Tuple{Bool, Int, UnitRange{Int}, Bool}
    end

    # Note from bvdmitri: see explanation about explicit `::Tuple{Bool, Int, UnitRange{Int}, Bool}` further in main filter procedure
    function __resolve_var_ref_position(
        qpair_name::Symbol,
        qpair_index::Union{Integer, FunctionalIndex, CombinedRange, SplittedRange},
        start_with::Int
    )::Tuple{Bool, Int, UnitRange{Int}, Bool}
        qpair_name_position = findnext(==(qpair_name), var_refs_names, start_with)
        if qpair_name_position === nothing
            return (false, 0, typemin(Int):typemax(Int), false)::Tuple{Bool, Int, UnitRange{Int}, Bool}
        end
        qpair_name_collection = @inbounds var_refs_collections[qpair_name_position]
        qpair_resolved_index  = __factorisation_specification_resolve_index(qpair_index, qpair_name_collection)
        if (@inbounds var_refs_indices[qpair_name_position]) ∈ qpair_resolved_index
            return (
                true,
                qpair_name_position,
                __as_unit_range(qpair_resolved_index),
                is_splitted(qpair_resolved_index)
            )::Tuple{Bool, Int, UnitRange{Int}, Bool}
        else
            return __resolve_var_ref_position(
                qpair_name,
                qpair_resolved_index,
                qpair_name_position + 1
            )::Tuple{Bool, Int, UnitRange{Int}, Bool}
        end
    end

    # `factorisation` is a tuple of `FactorisationConstraintsSpecification`s
    # FactorisationConstraintsSpecification has names of LHS and specs of RHS
    factorisation = constraints.factorisation

    function __process_factorisation_entry!(symbol::Symbol, index, shift::Int)
        # `symbols` refers to all possible symbols that refer to the current variable

        function __filter_template!(spec::FactorisationConstraintsSpecification, factorisation_entries::Tuple)
            # This function applies a given `spec` with rhs = `factorisation_entries`
            # Function goes all over `factorisation_entries` and check that the target `symbols` are found only once
            # This is to prevent situations like q(x) = q(x[1])q(x[1]), which are correct from syntax point of view, but are not allowed in runtime
            found_once = false
            for entry in factorisation_entries
                is_found = __filter_template!(Val(true), entry)
                if is_found && found_once
                    error(
                        "Found variable $(__repr_symbol_index(symbol, index)) twice in the factorisation specification $(spec)."
                    )
                end
                found_once = found_once | is_found
            end
            if !found_once
                error(
                    "Variable $(__repr_symbol_index(symbol, index)) has not been found on the RHS of the factorisation specification $(spec)"
                )
            end
            return found_once
        end

        # First argument `force` is a compile time flag that indicates if we want to check names of the `spec` first
        __filter_template!(force::Val{true}, spec::FactorisationConstraintsSpecification)  = __filter_template!(spec, getentries(spec))
        __filter_template!(force::Val{false}, spec::FactorisationConstraintsSpecification) = symbol ∈ getnames(spec) ? __filter_template!(spec, getentries(spec)) : false

        function __filter_template!(force::Val{true}, csentry::FactorisationConstraintsEntry)
            entry_names   = getnames(csentry)
            entry_indices = getindices(csentry)
            entry_pairs   = getpairs(csentry)

            # First, we check if current `symbol` is within `entry_names`.
            is_external_name::Bool = symbol ∉ entry_names
            # Sanity check if we actually found currnet `symbol` with the given `index`
            current_found::Bool = false
            # In case if we didn't find exact match, we go over all saved var_ref positions again and filter them out from the current cluster
            save_var_ref_position_tmp::Int = 0

            # Splitted entries require special care and runtime checks on bounds/diffs
            # We do checks every time though it is not strictly necessary (todo check once?)
            is_csentry_splitted::Bool = unrolled_all(index -> index isa SplittedRange, entry_indices)
            if is_csentry_splitted && length(entry_indices) >= 2
                split_first_indices = unrolled_map((ename, eindex) -> __factorisation_specification_resolve_index(firstindex(eindex), allvariables[ename]), entry_names, entry_indices)
                split_last_indices  = unrolled_map((ename, eindex) -> __factorisation_specification_resolve_index(lastindex(eindex), allvariables[ename]), entry_names, entry_indices)
                split_diff_indices  = unrolled_map(-, split_last_indices, split_first_indices)
                split_diff_check    = unrolled_all(==(first(split_diff_indices)), split_diff_indices)
                if !split_diff_check
                    error(
                        """Invalid splitted factorisation specification entry $(csentry). Indices difference in split expression should match. Evaluated to [ $(join(map((e1, e2) -> (e1:e2), split_first_indices, split_last_indices), ",")) ]"""
                    )
                end
            end

            # This is the main filter loop of the whole procedure and basically all the magic happens here
            # For each qpair from factorisation constraints
            #   - we find all matches between qpair and `var_refs` with the help of the `__resolve_var_ref_position` function
            #   - for each match we check if current `FactorisationConstraintsEntry` is `external` with the respect to the current `symbol`
            #       - in case of `external = true` we simply filter out everything
            #       - in case of `external = false` we track if we found exact match for current `symbol` and `index` and filter out only in case of splitted ranges
            unrolled_foreach(entry_pairs) do qpair
                q_pair_symbol    = qpair[1]
                q_pair_index     = qpair[2]
                var_ref_position = 0
                is_found         = true
                @inbounds while is_found
                    # Note from bvdmitri: There **were** some strange type-instability issues. I could resolve them only then I put this huge
                    # `::Tuple{Bool, Int, UnitRange{Int}, Bool}` type annotation. However in my head Julia should be able to resolve it automatically
                    # Nevertheless, it solves type-instability issues and makes x10 performance speedup for free?. Probably there is something else to improve,
                    # but overall performance is acceptable. (see also: `__resolve_var_ref_position` with explicit type annotation)
                    is_found, var_ref_position, var_ref_resolved_range, var_ref_is_splitted =
                        __resolve_var_ref_position(
                            q_pair_symbol,
                            q_pair_index,
                            var_ref_position + 1
                        )::Tuple{Bool, Int, UnitRange{Int}, Bool}
                    if is_found
                        var_ref_index = var_refs_indices[var_ref_position]
                        if is_external_name
                            clusters_template[shift+var_ref_position] = false
                        elseif q_pair_symbol === symbol && ((index === nothing) || (index ∈ var_ref_resolved_range))
                            if index === var_ref_index
                                current_found = true
                            elseif var_ref_is_splitted
                                clusters_template[shift+var_ref_position] = false
                            end
                        elseif q_pair_symbol !== symbol && is_csentry_splitted
                            # So this check is quite computationally expensive, but we assume it would happen rather rare
                            # We support it as a very special case (very handy though, imo, some extra computational overhead is reasonable here)

                            # this should not be `nothing` by any means
                            # this also should be unique since we don't allow multiple entries with the same name in splitted range
                            q_pair_index_for_current_symbol =
                                entry_indices[findnext(==(symbol), entry_names, 1)::Integer]

                            # So here we have
                            # `q_pair` for current entry and `q_pair for current symbol` which aren't the same because of the previous checks
                            # `var_ref_index` for current entry and `index` for current symbol which aren't the same because of the previous checks
                            # We need to check that diffs between `var_ref_index` and `index` is the same as between diffs `firstindex` of `q_pair`s
                            # If not, we filter out `clusters_template`
                            q_pair_diff = firstindex(__factorisation_specification_resolve_index(q_pair_index, allvariables[q_pair_symbol])) - firstindex(__factorisation_specification_resolve_index(q_pair_index_for_current_symbol, allvariables[symbol]))
                            index_diff  = var_ref_index - index

                            if q_pair_diff !== index_diff
                                clusters_template[shift+var_ref_position] = false
                            end
                        else
                            save_var_ref_position_tmp += 1
                            var_refs_positions[save_var_ref_position_tmp] = var_ref_position
                        end
                    else
                        break
                    end
                end
            end

            if !is_external_name && !current_found
                @inbounds for i in 1:save_var_ref_position_tmp
                    clusters_template[shift+var_refs_positions[i]] = false
                end
            end

            return current_found
        end

        unrolled_foreach(factorisation) do spec
            __filter_template!(Val(false), spec)
        end
    end

    index::Int = 1
    shift::Int = 0
    for varref in var_refs
        if israndom(varref[3])
            # We process everything as usual if varref is a random variable
            __process_factorisation_entry!(varref[1], varref[2], shift)
        else
            # We filter out varref from all clusters if it is not random
            for k in 1:N
                if k !== index
                    clusters_template[shift+k] = false
                    clusters_template[(k-1)*N+index] = false
                end
            end
        end
        index += 1
        shift += N
    end

    # In this last step we transform templates from `clusters_template` into a set of tuples
    @inbounds for index in 1:N
        range_left  = (index - 1) * N + 1
        range_right = range_left + N - 1

        ki = 0
        @inbounds for (index, flag) in enumerate(view(clusters_template, range_left:range_right))
            if flag
                ki += 1
                cluster_indices[ki] = index
            end
        end

        output = Tuple(view(cluster_indices, 1:ki))

        push!(clusters_set, output)
    end

    # ReactiveMP backend assumes clusters are sorted by first index
    sorted_clusters = sort!(collect(clusters_set); by = first, alg = QuickSort)

    # Check if clusters do intersect
    for cluster in sorted_clusters
        for index in cluster
            if clusters_usage[index] === true
                __throw_intersection_error(fform, var_refs, var_refs_names, sorted_clusters, constraints)
            end
            clusters_usage[index] = true
        end
    end

    return Tuple(sorted_clusters)
end

## Errors

struct ClusterIntersectionError
    fform
    varrefs
    varrefsnames
    clusters
    constraints
end

__throw_intersection_error(fform, varrefs, varrefsnames, clusters, constraints) =
    throw(ClusterIntersectionError(fform, varrefs, varrefsnames, clusters, constraints))

function Base.showerror(io::IO, error::ClusterIntersectionError)
    print(
        io,
        "Cluster intersection error in the expression `$(__io_entry_pair(error.varrefs[1])) ~ $(error.fform)($(join(map(__io_entry_pair, error.varrefs[2:end]), ", ")))`.\n"
    )
    print(io, "Based on factorisation constraints the resulting local constraint ")
    print(io, "q(")
    join(io, map(r -> __io_entry_pair(r[1], r[2]), error.varrefs), ", ")
    print(io, ") = ")
    for cluster in error.clusters
        print(io, "q(")
        entries = map(cluster) do clusterindex
            __io_entry_pair(error.varrefs[clusterindex])
        end
        join(io, entries, ", ")
        print(io, ")")
    end
    print(io, " has cluster intersections, which is disallowed by default.")
    print(io, "\nTechnical info: clusters = ", error.clusters)
    print(io, "\n", error.constraints)
end
