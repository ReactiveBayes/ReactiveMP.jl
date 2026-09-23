const SELECTED_RULES = IdDict{Any, Vector{LineNumberNode}}()

record_selected_rule!(spec, source::LineNumberNode) = (push!(get!(SELECTED_RULES, spec, LineNumberNode[]), source); nothing)
