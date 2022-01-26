
struct Constraints 
    nodes :: Dict{FactorisationSpec, FactorisationSpecNode}

    Constraints() = new(Dict{FactorisationSpec, FactorisationSpecNode}())
end


make_factorisation_spec_entry(constraints::Constraints, args...) = FactorisationSpecEntry(args...)

function search_factorisation_spec(constraints::Constraints, args::FactorisationSpecEntry...) 
    spec = FactorisationSpec(args)
    if haskey(constraints.nodes, spec)
        return constraints.nodes[spec]
    else
        return spec
    end
end

function add_factorisation_node(constraints::Constraints, key::FactorisationSpec, entries::Union{ FactorisationSpec, FactorisationSpecNode }...)
    if haskey(constraints.nodes, key)
        error("Factorisation spec for $(key) exists already.")
    end
    childnodes = filter(d -> d isa FactorisationSpecNode, entries)
    childspec  = filter(d -> d isa FactorisationSpec, entries)
    node = FactorisationSpecNode(key, childnodes, childspec)
    constraints.nodes[key] = node
    return constraints
end
