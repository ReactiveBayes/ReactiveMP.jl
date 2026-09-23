# Which v6 rules the Phase 4.5 slice models actually run: every message rule, marginal rule
# and average energy selected while recording the engine fixtures, with the v6 method it
# resolved to. This is the porting list for the slice.
#
#   julia --startup-file=no --project=compat/v6-comparison compat/v6-comparison/slice_rule_inventory.jl
#
# Message rules are found from the recorder's rule-call events, by asking `which` with the
# arguments v6 dispatched on (`message.jl:692-703`). Marginal rules and average energies
# emit no event, so this script, and only this script, redefines the two v6 call sites that
# make them — `MarginalMapping`'s callable and the stochastic-node free-energy score — with
# their bodies copied unchanged apart from the logging line. Line coverage cannot replace
# this: a one-line `@rule … = expr` is attributed to the macro, not to the rule file.

include(joinpath(@__DIR__, "record_engine_fixtures.jl"))

using ReactiveMP: MarginalMapping, MessageMapping, AverageEnergy, FactorBoundFreeEnergy, Stochastic, AbstractFactorNode,
    getrecent, is_clamped, is_clamped_or_initial, marginal_mapping_fform, message_mapping_fform, marginalrule,
    get_node_local_marginals, getlocalclusters, get_stream_of_marginals, skip_initial, functionalform, DifferentialEntropy,
    postprocess_stream_of_scores, __check_all, typeofdata
# Not direct dependencies of this environment; reached through ReactiveMP.
const Rocket = ReactiveMP.Rocket
const TupleTools = ReactiveMP.TupleTools
using .Rocket: combineLatest, PushNew

const USED = Dict{Tuple, Int}()
const CURRENT = Ref("")

# A macro-generated method reports the macro's own line, not the rule's, so a selected rule
# is identified by the input types it declares: the element types of its signature's message
# and marginal tuples.
function declared_type(p)
    body = Base.unwrap_unionall(p)
    (body isa DataType && body.name.wrapper in (Marginal, ReactiveMP.Message) && !isempty(body.parameters)) || return p
    parameter = body.parameters[1]
    return parameter isa TypeVar ? parameter.ub : parameter
end
declared_types(t) = t === Nothing ? () : t isa DataType && t <: Tuple ? map(p -> string(declared_type(p)), Tuple(t.parameters)) : ("$t",)
function location(m::Method, positions)
    sig = Base.unwrap_unionall(m.sig)
    return join((join(declared_types(sig.parameters[i]), ", ") for i in positions if i <= length(sig.parameters)), "; ")
end
types_text(names, values) = values === nothing ? "" : join(("$n::$(nameof(typeofdata(v)))" for (n, v) in zip(names, values)), ", ")
unval(::Val{T}) where {T} = T
unval(::Nothing) = ()

function note!(kind, node, target, inputs, method)
    key = (CURRENT[], kind, node, target, inputs, method)
    USED[key] = get(USED, key, 0) + 1
    return nothing
end

RULE_CALL_HOOK[] = function (event)
    mapping = event.mapping
    # A call with a missing input never reaches a rule (`message.jl:684-690`).
    for inputs in (event.messages, event.marginals)
        (inputs !== nothing && any(m -> ismissing(ReactiveMP.getdata(m)), inputs)) && return nothing
    end
    args = (
        message_mapping_fform(mapping), mapping.vtag, mapping.vconstraint, mapping.msgs_names, event.messages,
        mapping.marginals_names, event.marginals, mapping.meta, event.annotations, mapping.factornode,
    )
    method = location(which(ReactiveMP.rule, map(Core.Typeof, args)), (6, 8))
    inputs = join(filter(!isempty, [types_text(map(n -> "m_$n", unval(mapping.msgs_names)), event.messages), types_text(map(n -> "q_$n", unval(mapping.marginals_names)), event.marginals)]), ", ")
    return note!(:message, node_text(mapping), target_text(mapping.vtag), inputs, method)
end

# v6 `src/marginal.jl:283-318`, plus the logging line.
function (mapping::MarginalMapping)(dependencies)
    messages = getrecent(dependencies[1])
    marginals = getrecent(dependencies[2])
    is_marginal_clamped = __check_all(is_clamped, messages) && __check_all(is_clamped, marginals)
    is_marginal_initial = !is_marginal_clamped && (__check_all(is_clamped_or_initial, messages) && __check_all(is_clamped_or_initial, marginals))
    marginal = if !isnothing(messages) && any(ismissing, TupleTools.flatten(getdata.(messages)))
        missing
    elseif !isnothing(marginals) && any(ismissing, TupleTools.flatten(getdata.(marginals)))
        missing
    else
        args = (marginal_mapping_fform(mapping), mapping.vtag, mapping.msgs_names, messages, mapping.marginals_names, marginals, mapping.meta, mapping.factornode)
        inputs = join(filter(!isempty, [types_text(map(n -> "m_$n", unval(mapping.msgs_names)), messages), types_text(map(n -> "q_$n", unval(mapping.marginals_names)), marginals)]), ", ")
        note!(:marginal, string(nameof(marginal_mapping_fform(mapping))), string(mapping.vtag), inputs, location(which(marginalrule, map(Core.Typeof, args)), (5, 7)))
        marginalrule(args...)
    end
    return Marginal(marginal, is_marginal_clamped, is_marginal_initial)
end

# v6 `src/score/node.jl:80-112`, plus the logging line.
function ReactiveMP.score(::Type{T}, ::FactorBoundFreeEnergy, ::Stochastic, node::AbstractFactorNode, meta, stream_postprocessors) where {T <: ReactiveMP.CountingReal}
    fnstream = (localmarginal) -> get_stream_of_marginals(localmarginal) |> skip_initial()
    localmarginals = get_node_local_marginals(getlocalclusters(node))
    stream = combineLatest(map(fnstream, localmarginals), PushNew())
    mapping = let fform = functionalform(node), marginal_names = Val{Tuple(map(ReactiveMP.name, localmarginals))}()
        (marginals) -> begin
            names = unval(marginal_names)
            args = (AverageEnergy(), fform, marginal_names, marginals, meta)
            note!(:average_energy, string(nameof(fform)), string(names), types_text(map(n -> "q_$n", names), marginals), location(which(ReactiveMP.score, map(Core.Typeof, args)), (5,)))
            average_energy = ReactiveMP.score(args...)
            clusters_entropy = mapreduce(marginal -> ReactiveMP.score(DifferentialEntropy(), marginal), +, marginals)
            return convert(T, average_energy - clusters_entropy)
        end
    end
    return postprocess_stream_of_scores(stream_postprocessors, stream |> Rocket.map(T, mapping))
end

for (id, _, run) in MODELS
    CURRENT[] = id
    run()
end

println("| model | kind | node | target | inputs | the v6 rule's declared types (messages; marginals) | calls |")
println("|---|---|---|---|---|---|---|")
for key in sort!(collect(keys(USED)))
    model, kind, node, target, inputs, loc = key
    println("| $model | $kind | $node | `$target` | `$inputs` | `$loc` | $(USED[key]) |")
end
