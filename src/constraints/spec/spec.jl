
import Base: show

struct ConstraintsGenerator{F}
    generator :: F
end

Base.show(io::IO, generator::ConstraintsGenerator) = print(io, "ConstraintsGenerator()")

function (constraints::ConstraintsGenerator)(model)
    factorisation = constraints.generator(model)
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
