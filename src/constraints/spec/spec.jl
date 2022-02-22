export ConstraintsSpecification

import Base: show

const ConstraintsSpecificationPreallocatedDefaultSize = 64

struct ConstraintsSpecificationPreallocated
    clusters_template   :: BitVector
    clusters_usage      :: BitVector
    clusters_set        :: Set{Tuple}
    cluster_indices     :: Vector{Int}
    var_refs_positions  :: Vector{Int}
    
    ConstraintsSpecificationPreallocated() = new(trues(ConstraintsSpecificationPreallocatedDefaultSize), falses(ConstraintsSpecificationPreallocatedDefaultSize), Set{Tuple}(), Vector{Int}(undef, ConstraintsSpecificationPreallocatedDefaultSize), Vector{Int}(undef, ConstraintsSpecificationPreallocatedDefaultSize))
end

function __reset_preallocated!(preallocated::ConstraintsSpecificationPreallocated, size::Int)
    abs2size = abs2(size)
    if length(preallocated.clusters_template) < abs2size
        resize!(preallocated.clusters_template, abs2size)
        resize!(preallocated.clusters_usage, abs2size) # note: we dont need `size^2` for this, just `size` should be enough, but just to avoid extra checks
        resize!(preallocated.var_refs_positions, abs2size)
    end
    
    if length(preallocated.cluster_indices) < size
        resize!(preallocated.cluster_indices, size)
    end
    
    fill!(preallocated.clusters_template, true)
    fill!(preallocated.clusters_usage, false)
    
    empty!(preallocated.clusters_set)
end

struct ConstraintsSpecification{F, M}
    factorisation :: F
    form :: M
    preallocated :: ConstraintsSpecificationPreallocated
end

ConstraintsSpecification(factorisation::F, form::M) where { F, M } = ConstraintsSpecification{F, M}(factorisation, form, ConstraintsSpecificationPreallocated())

__reset_preallocated!(specification::ConstraintsSpecification, size::Int) = __reset_preallocated!(specification.preallocated, size)

function Base.show(io::IO, specification::ConstraintsSpecification) 
    print(io, "Constraints:\n\tform: $(specification.form)\n")
    print(io, "\tfactorisation\n")
    foreach(specification.factorisation) do f
        print(io, "\t\t", f, "\n")
    end
end

"""
    resolve_factorisation(expr::Expr, variables, constraints, model) 

This function resolves factorisation constraints in a form of a tuple for a given `expr` (needed for error printing), `variables`, `constraints` and `model`.

See also: [`ConstraintsSpecification`](@ref)
"""
function resolve_factorisation(expr::Expr, variables, constraints, model) 

    N = length(variables)
    
    preallocated = constraints.preallocated
    
    __reset_preallocated!(preallocated, N)
    
    clusters_template   = preallocated.clusters_template
    clusters_usage      = preallocated.clusters_usage
    clusters_set        = preallocated.clusters_set
    cluster_indices     = preallocated.cluster_indices
    var_refs_positions  = preallocated.var_refs_positions
    
    var_refs = map(get_factorisation_reference, variables)
    
    var_refs_names       = map(r -> r[1], var_refs)
    var_refs_indices     = map(r -> r[2], var_refs)
    
    model_vardict = ReactiveMP.getvardict(model)
    
    var_refs_collections = map(var_refs_names) do name
        return get(() -> error("Model has no variable named $(name). Or it has not been created before the expression `$(expr)`."), model_vardict, name)
    end
    
    # function _resolve_var_ref_position
    #     return is_found, index, range, is_splitted
    # end
    
    # Note from bvdmitri: see explanation about explicit `::Tuple{Bool, Int, UnitRange{Int}, Bool}` further in main filter procedure
    function __resolve_var_ref_position(qpair_name::Symbol, qpair_index::Nothing, start_with::Int)::Tuple{Bool, Int, UnitRange{Int}, Bool}
        position    = findnext(==(qpair_name), var_refs_names, start_with)
        is_found    = position !== nothing
        _position   = is_found ? position : 0
        range       = typemin(Int):typemax(Int)
        is_splitted = false
        return (is_found, _position, range, is_splitted)::Tuple{Bool, Int, UnitRange{Int}, Bool}
    end
    
    # Note from bvdmitri: see explanation about explicit `::Tuple{Bool, Int, UnitRange{Int}, Bool}` further in main filter procedure
    function __resolve_var_ref_position(qpair_name::Symbol, qpair_index::Union{Integer, FunctionalIndex, CombinedRange, SplittedRange}, start_with::Int)::Tuple{Bool, Int, UnitRange{Int}, Bool}
        qpair_name_position = findnext(==(qpair_name), var_refs_names, start_with)
        if qpair_name_position === nothing
            return (false, 0, typemin(Int):typemax(Int), false)::Tuple{Bool, Int, UnitRange{Int}, Bool}
        end
        qpair_name_collection = @inbounds var_refs_collections[qpair_name_position]
        qpair_resolved_index  = __factorisation_specification_resolve_index(qpair_index, qpair_name_collection)
        if (@inbounds var_refs_indices[qpair_name_position]) ∈ qpair_resolved_index
            return (true, qpair_name_position, __as_unit_range(qpair_resolved_index), is_splitted(qpair_resolved_index))::Tuple{Bool, Int, UnitRange{Int}, Bool}
        else
            return __resolve_var_ref_position(qpair_name, qpair_resolved_index, qpair_name_position + 1)::Tuple{Bool, Int, UnitRange{Int}, Bool}
        end
    end
    
    # `factorisation` is a tuple of `FactorisationConstraintsSpecification`s
    # FactorisationConstraintsSpecification has names of LHS and specs of RHS
    factorisation = constraints.factorisation
    
    function __process_factorisation_entry!(symbol::Symbol, index, shift::Int) where N
        # `symbols` refers to all possible symbols that refer to the current variable
        
        function __filter_template!(spec::FactorisationConstraintsSpecification, factorisation_entries::Tuple)
            # This function applies a given `spec` with rhs = `factorisation_entries`
            # Function goes all over `factorisation_entries` and check that the target `symbols` are found only once
            # This is to prevent situations like q(x) = q(x[1])q(x[1]), which are correct from syntax point of view, but are not allowed in runtime
            found_once = false
            for entry in factorisation_entries
                is_found = __filter_template!(Val(true), entry)
                if is_found && found_once
                    error("Found variable $(__repr_symbol_index(symbol, index)) twice in the factorisation specification $(spec).")
                end
                found_once = found_once | is_found
            end
            if !found_once
                 error("Variable $(__repr_symbol_index(symbol, index)) has not been found on the RHS of the factorisation specification $(spec)")
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
                split_first_indices = unrolled_map((ename, eindex) -> __factorisation_specification_resolve_index(firstindex(eindex), model[ename]), entry_names, entry_indices)
                split_last_indices  = unrolled_map((ename, eindex) -> __factorisation_specification_resolve_index(lastindex(eindex), model[ename]), entry_names, entry_indices)
                split_diff_indices  = unrolled_map(-, split_last_indices, split_first_indices)
                split_diff_check    = unrolled_all(==(first(split_diff_indices)), split_diff_indices)
                if !split_diff_check
                     error("""Invalid splitted factorisation specification entry $(csentry). Indices difference in split expression should match. Evaluated to [ $(join(map((e1, e2) -> (e1:e2), split_first_indices, split_last_indices), ",")) ]""")
                end
            end
        
            # This is the main filter loop of the whole procedure and basically all the magic happens here
            # For each qpair from factorisation constraints
            #   - we find all matches between qpair and `var_refs` with the help of the `__resolve_var_ref_position` function
            #   - for each match we check if current `FactorisationConstraintsEntry` is `external` with the respect to the current `symbol`
            #       - in case of `external = true` we simply filter out everything
            #       - in case of `external = false` we track if we found exact match for current `symbol` and `index` and filter out only in case of splitted ranges
            unrolled_foreach(entry_pairs) do qpair
                q_pair_symbol = qpair[1]
                q_pair_index  = qpair[2]
                var_ref_position = 0
                is_found         = true
                @inbounds while is_found
                    # Note from bvdmitri: There **were** some strange type-instability issues. I could resolve them only then I put this huge
                    # `::Tuple{Bool, Int, UnitRange{Int}, Bool}` type annotation. However in my head Julia should be able to resolve it automatically
                    # Nevertheless, it solves type-instability issues and makes x10 performance speedup for free?. Probably there is something else to improve,
                    # but overall performance is acceptable. (see also: `__resolve_var_ref_position` with explicit type annotation)
                    is_found, var_ref_position, var_ref_resolved_range, var_ref_is_splitted = __resolve_var_ref_position(q_pair_symbol, q_pair_index, var_ref_position + 1)::Tuple{Bool, Int, UnitRange{Int}, Bool}                    
                    if is_found
                        var_ref_index = var_refs_indices[var_ref_position]
                        if is_external_name
                            clusters_template[ shift + var_ref_position ] = false
                        elseif q_pair_symbol === symbol && ((index === nothing) || (index ∈ var_ref_resolved_range))
                            if index === var_ref_index
                                current_found = true
                            elseif var_ref_is_splitted
                                clusters_template[ shift + var_ref_position ] = false
                            end
                        elseif q_pair_symbol !== symbol && is_csentry_splitted
                            # So this check is quite computationally expensive, but we assume it would happen rather rare
                            # We support it as a very special case (very handy though, imo, some extra computational overhead is reasonable here)
                            
                            # this should not be `nothing` by any means
                            # this also should be unique since we don't allow multiple entries with the same name in splitted range
                            q_pair_index_for_current_symbol = entry_indices[ findnext(==(symbol), entry_names, 1)::Integer ] 
                            
                            # So here we have
                            # `q_pair` for current entry and `q_pair for current symbol` which aren't the same because of the previous checks
                            # `var_ref_index` for current entry and `index` for current symbol which aren't the same because of the previous checks
                            # We need to check that diffs between `var_ref_index` and `index` is the same as between diffs `firstindex` of `q_pair`s
                            # If not, we filter out `clusters_template`
                            q_pair_diff = firstindex(__factorisation_specification_resolve_index(q_pair_index, model[q_pair_symbol])) - firstindex(__factorisation_specification_resolve_index(q_pair_index_for_current_symbol, model[symbol]))
                            index_diff  = var_ref_index - index
                            
                            if q_pair_diff !== index_diff
                                clusters_template[ shift + var_ref_position ] = false
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
                    clusters_template[ shift + var_refs_positions[i] ] = false
                end
            end
            
            return current_found
        end
        
        unrolled_foreach(factorisation) do spec
            __filter_template!(Val(false), spec)
        end
        
    end
    
    shift::Int = 0
    for varref in var_refs
        __process_factorisation_entry!(varref[1], varref[2], shift)
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
                __throw_intersection_error(expr, var_refs, sorted_clusters, constraints)
            end 
            clusters_usage[index] = true
        end
    end
    
    return Tuple(sorted_clusters)
end