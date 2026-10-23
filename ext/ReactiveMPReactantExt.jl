module ReactiveMPReactantExt

using ReactiveMP, Reactant

# Why this extension exists. Reactant compiles a function by tracing it: it runs the function once,
# rewriting every call inside it to Reactant's overlays. A model compiled with Reactant therefore has
# its whole factor graph built and wired under tracing. Since Reactant 0.2.292
# (EnzymeAD/Reactant.jl#3378), the overlays of `all` and `any` decide whether to trace by walking every
# field of every object in their arguments. A graph's variables hold Rocket's linked lists, whose
# placeholder nodes have undefined fields and whose nodes point at each other, so the walk throws an
# `UndefRefError` or never ends, and compiling a model of more than one step fails.
#
# The functions marked below build the graph or tell a variable's kind: they never compute with a
# variable's value, so they need no overlay and run as plain Julia under tracing. The rules, which do
# compute with the values, are not marked and stay traced. `factornode` builds a node, its interfaces,
# clusters and node function, and `create_factornode` does the work: a call such as
# `factornode(fform, interfaces, factorisation)` is inlined into its caller, so Reactant never sees
# `factornode` itself. `israndom`, `isdata` and `isconst` tell the kind of one variable or of a
# collection of them. `group_length` counts a group's interfaces; tracing its `count` would walk
# their streams even though only their types and names matter. The marks can go once Reactant's
# walk tolerates undefined fields and cycles
# (EnzymeAD/Reactant.jl#3506).
#
# The marks go into a set in Reactant, which precompilation does not keep, so they are made when the
# extension loads.
function __init__()
    Reactant.@skip_rewrite_func ReactiveMP.factornode
    Reactant.@skip_rewrite_func ReactiveMP.create_factornode
    Reactant.@skip_rewrite_func ReactiveMP.israndom
    Reactant.@skip_rewrite_func ReactiveMP.isdata
    Reactant.@skip_rewrite_func ReactiveMP.isconst
    Reactant.@skip_rewrite_func ReactiveMP.group_length
    return nothing
end

end
