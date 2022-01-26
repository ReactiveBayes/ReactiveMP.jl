
import Base: show

struct Constraints 
    factorisation :: Dict{FactorisationSpec, FactorisationSpecNode}

    Constraints() = new(Dict{FactorisationSpec, FactorisationSpecNode}())
end

function Base.show(io::IO, constraints::Constraints)
    for (key, value) in constraints.factorisation
        println(io, key, " => ", value)
    end
end

make_factorisation_spec_entry(constraints::Constraints, args...)                   = FactorisationSpecEntry(args...)
make_factorisation_spec(constraints::Constraints, args::FactorisationSpecEntry...) = FactorisationSpec(args)

add_factorisation_node(constraints::Constraints, key::FactorisationSpec, entries::FactorisationSpec) = add_factorisation_node(constraints, key, (entries, ))

function add_factorisation_node(constraints::Constraints, key::FactorisationSpec, entries::NTuple{N, FactorisationSpec}) where N
    !(haskey(constraints.factorisation, key)) || error("Factorisation spec for $(key) exists already.")
    node = FactorisationSpecNode(entries)
    constraints.factorisation[key] = node
    return constraints
end
