
import Base: show

struct ConstraintsGenerator{F}
    generator :: F
end

Base.show(io::IO, generator::ConstraintsGenerator) = print(io, "ConstraintsGenerator()")

function (constraints::ConstraintsGenerator)(model)
    factorisation = constraints.generator(model)

    vardict = getvardict(model)
    names   = keys(getvardict(model))

    # Sanity check
    for (spec, list) in factorisation
        # First check that on LHS of factorisation constraints expression all variables present in the model
        validate(spec, vardict, names, allow_dots = true)
        # check lhs and rhs nams 
        lhs_names = Iterators.map(name, getentries(spec))
        rhs_names = Iterators.map(name, TupleTools.flatten(map(getentries, getentries(list))))
        for lname in lhs_names
            if lname !== :(..) && lname ∉ rhs_names
                error("Error in $(spec) = $(list) factorisation specification. Variable $(lname) is present on the left side of the expression, but not used on the right side.")
            end
        end
        # Next check the same but for all entries on RHS
        for entry in getentries(list)
            validate(entry, vardict, names, allow_dots = false)
        end
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

function add_factorisation_node(factorisation::Dict{FactorisationSpec, FactorisationSpecList}, key::FactorisationSpec, entries::FactorisationSpec) 
    return add_factorisation_node(factorisation, key, (entries, ))
end

function add_factorisation_node(factorisation::Dict{FactorisationSpec, FactorisationSpecList}, key::FactorisationSpec, entries::NTuple{N, FactorisationSpec}) where N
    !(haskey(factorisation, key)) || error("Factorisation spec for $(key) exists already.")
    node = FactorisationSpecList(entries)
    factorisation[key] = node
    return factorisation
end
