
import Base: show

struct ConstraintsGenerator{F}
    generator :: F
end

Base.show(io::IO, ::ConstraintsGenerator) = print(io, "ConstraintsGenerator()")

function (constraints::ConstraintsGenerator)(model)
    factorisation = constraints.generator(model)

    # Check here that root exists
    haskey(factorisation, RootFactorisationSpec) || error("Factorisation constraints specification must has `q(..) = ` expression as a root expression.")

    vardict = getvardict(model)
    names   = keys(getvardict(model))

    # Initial sanity check, more checks will occur on actual constraints-to-model application
    for (spec, list) in factorisation
        # First check that on LHS of factorisation constraints expression all variables present in the model
        validate(spec, vardict, names, allow_dots = true)
        # Next check the same but for all entries on RHS
        foreach(entry -> validate(entry, vardict, names, allow_dots = false), getentries(list))
        # Next, we check that LHS and RHS names matched and they're indices match too and if indices did intersect
        spec_entries = getentries(spec)
        flat_list = TupleTools.flatten(map(getentries, getentries(list)))
        # Here in case of `q(..)` we replace lhs with `q(args...)` where args are names from right expression
        rhs_check_entries = if length(spec_entries) === 1 && name(first(spec_entries)) === :(..)
            Set(map(e -> FactorisationSpecEntry(name(e), nothing), flat_list))
        else 
            spec_entries
        end
        for spec_entry in rhs_check_entries
            spec_name     = name(spec_entry)
            if spec_name !== :(..)
                filtered_list = Iterators.filter(entry -> name(entry) === spec_name, flat_list)
                processed = false # `processed` = false means that there are no variables with the same name on the LHS
                maxindex = typemin(Int64)
                minindex = typemax(Int64)
                for value in filtered_list
                    vfirstindex = firstindex(value)
                    vlastindex  = lastindex(value)
                    if length(intersect(minindex:maxindex, vfirstindex:vlastindex)) !== 0
                        error("Error in $(spec) = $(list) factorisation constraint specification. Indices for variable $(spec_name) intersect: $(minindex:maxindex) ∩ $(vfirstindex:vlastindex) != ∅")
                    end
                    minindex = if vfirstindex < minindex vfirstindex else minindex end
                    maxindex = if vlastindex > maxindex vlastindex else maxindex end
                    processed = true
                end
                if !processed
                    error("Error in $(spec) = $(list) factorisation constraint specification. Variable $(spec_name) is present on the left side of the expression, but not used on the right side.")
                else
                    minindex = if minindex === typemin(Int64) firstindex(model, spec_name) else minindex end
                    maxindex = if maxindex === typemax(Int64) lastindex(model, spec_name) else maxindex end
                    spec_entry_minindex = firstindex(spec_entry)
                    spec_entry_maxindex = lastindex(spec_entry)
                    spec_entry_minindex = if spec_entry_minindex === typemin(Int64) firstindex(model, spec_name) else spec_entry_minindex end
                    spec_entry_maxindex = if spec_entry_maxindex === typemax(Int64) lastindex(model, spec_name) else spec_entry_maxindex end
                    if spec_entry_minindex !== minindex || spec_entry_maxindex !== maxindex
                        error("Error in $(spec) = $(list) factorisation specification. Indices for $(name(spec_entry)) on the left hand side of the expression and on the right hand side do not match.")
                    end
                end
            end
        end
    end

    # Check here that all factorisation specification paths are used and `error` if some is unused
    root = factorisation[ RootFactorisationSpec ]
    used = Set{FactorisationSpec}()

    function __util_step_into(used::Set, factorisation::Dict, entries::Tuple)
        for entry in entries
            if entry ∈ used
                error("Factorisation constraint specification $(entry) has been used twice.")
            elseif haskey(factorisation, entry)
                push!(used, entry)
                __util_step_into(used, factorisation, getentries(factorisation[ entry ]))
            end
        end
    end

    __util_step_into(used, factorisation, getentries(root))

    unused = setdiff(keys(factorisation), used, (RootFactorisationSpec, ))
    if length(unused) !== 0
        error("Unused factorisation specification found: $(first(unused))")
    end

    return Constraints(factorisation)
end

struct Constraints
    factorisation :: Dict{FactorisationSpec, FactorisationSpecList}
end

function Base.show(io::IO, constraints::Constraints)
    for (key, value) in constraints.factorisation
        println(io, key, " => ", value)
    end
end

function add_factorisation_spec_list(factorisation::Dict{FactorisationSpec, FactorisationSpecList}, key::FactorisationSpec, entries::FactorisationSpec) 
    return add_factorisation_spec_list(factorisation, key, (entries, ))
end

function add_factorisation_spec_list(factorisation::Dict{FactorisationSpec, FactorisationSpecList}, key::FactorisationSpec, entries::NTuple{N, FactorisationSpec}) where N
    !(haskey(factorisation, key)) || error("Factorisation spec for $(key) exists already.")
    node = FactorisationSpecList(entries)
    factorisation[key] = node
    return factorisation
end
