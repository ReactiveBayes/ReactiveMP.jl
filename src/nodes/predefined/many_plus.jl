export ManyPlus

"""
    ManyPlus

A deterministic factor that adds a collection of two or more scalar Gaussian
inputs with a single factor node, without introducing intermediate sum variables.

The node uses sum-product messages on all edges, independently of the surrounding
factorisation. It supports univariate Gaussian messages in all parameterisations
and computes its contribution to the Bethe free energy.

Use it in an RxInfer model as

```julia
total := ManyPlus(inputs = [x1, x2, x3])
```
"""
struct ManyPlus end

as_node_symbol(::Type{ManyPlus}) = :ManyPlus
interfaces(::Type{ManyPlus}) = Val((:out, :inputs))
inputinterfaces(::Type{ManyPlus}) = Val((:inputs,))
alias_interface(::Type{ManyPlus}, ::Int64, name::Symbol) = name
is_predefined_node(::Type{ManyPlus}) = PredefinedNodeFunctionalForm()
sdtype(::Type{ManyPlus}) = Deterministic()

struct ManyPlusNodeFactorisation end

collect_factorisation(::Type{ManyPlus}, factorisation) =
    ManyPlusNodeFactorisation()

struct ManyPlusFactorNode{N} <: AbstractFactorNode
    out::NodeInterface
    inputs::NTuple{N, IndexedNodeInterface}
end

functionalform(::ManyPlusFactorNode) = ManyPlus
getinterfaces(node::ManyPlusFactorNode) = (node.out, node.inputs...)
getinboundinterfaces(node::ManyPlusFactorNode) = node.inputs
sdtype(::ManyPlusFactorNode) = Deterministic()

interfaceindices(node::ManyPlusFactorNode, name::Symbol) =
    (interfaceindex(node, name),)
interfaceindices(node::ManyPlusFactorNode, names::NTuple{N, Symbol}) where {N} =
    map(name -> interfaceindex(node, name), names)

function interfaceindex(node::ManyPlusFactorNode, name::Symbol)
    if name === :out
        return 1
    elseif name === :inputs
        return 2
    end

    error(
        "Unknown interface ':$(name)' for the [ $(functionalform(node)) ] node"
    )
end

function factornode(::Type{ManyPlus}, interfaces, factorisation)
    out_index = findfirst(interface -> first(interface) === :out, interfaces)
    isnothing(out_index) &&
        throw(ArgumentError("`ManyPlus` requires an `out` interface."))

    input_interfaces = filter(
        interface -> first(interface) === :inputs, interfaces
    )
    ninputs = length(input_interfaces)

    ninputs >= 2 || throw(
        ArgumentError(
            "`ManyPlus` requires at least two inputs; got $(ninputs)."
        ),
    )

    return ManyPlusFactorNode(
        NodeInterface(interfaces[out_index]...),
        ntuple(
            index -> IndexedNodeInterface(
                index, NodeInterface(input_interfaces[index]...)
            ),
            ninputs,
        ),
    )
end

struct ManyPlusFunctionalDependencies <: FunctionalDependencies end

collect_functional_dependencies(::ManyPlusFactorNode, ::Nothing) =
    ManyPlusFunctionalDependencies()
collect_functional_dependencies(
    ::ManyPlusFactorNode, ::ManyPlusFunctionalDependencies
) = ManyPlusFunctionalDependencies()
collect_functional_dependencies(::ManyPlusFactorNode, dependencies) = error(
    "The functional dependencies for `ManyPlus` must be `nothing` or " *
    "`ManyPlusFunctionalDependencies`, got `$(typeof(dependencies))`.",
)

function activate!(
    node::ManyPlusFactorNode, options::FactorNodeActivationOptions
)
    dependencies = collect_functional_dependencies(
        node, getdependecies(options)
    )
    return activate!(dependencies, node, options)
end

function functional_dependencies(
    ::ManyPlusFunctionalDependencies,
    node::ManyPlusFactorNode{N},
    interface,
    interface_index::Int,
) where {N}
    message_dependencies = if interface_index === 1
        (node.inputs,)
    elseif 2 <= interface_index <= N + 1
        target_index = interface_index - 1
        other_inputs = ntuple(N - 1) do index
            node.inputs[index < target_index ? index : index + 1]
        end
        (node.out, other_inputs)
    else
        error("Bad interface index $(interface_index) for `ManyPlus`.")
    end

    return message_dependencies, ()
end

function collect_latest_messages(
    ::ManyPlusFunctionalDependencies,
    ::ManyPlusFactorNode{N},
    dependencies::Tuple{NTuple{N, IndexedNodeInterface}},
) where {N}
    inputs = dependencies[1]
    streams = map(get_stream_of_inbound_messages, inputs)

    names = Val{(:inputs,)}()
    observable = combineLatest(streams, PushNew()) |> map_to((ManyOf(streams),))

    return names, observable
end

function collect_latest_messages(
    ::ManyPlusFunctionalDependencies,
    ::ManyPlusFactorNode,
    dependencies::Tuple{NodeInterface, NTuple{N, IndexedNodeInterface}},
) where {N}
    out = dependencies[1]
    inputs = dependencies[2]
    out_stream = get_stream_of_inbound_messages(out)
    input_streams = map(get_stream_of_inbound_messages, inputs)

    names = Val{(:out, :inputs)}()
    observable =
        combineLatest(
            (out_stream, combineLatest(input_streams, PushNew())), PushNew()
        ) |> map_to((out_stream, ManyOf(input_streams)))

    return names, observable
end

collect_latest_marginals(
    ::ManyPlusFunctionalDependencies, ::ManyPlusFactorNode, ::Tuple{}
) = (nothing, of(nothing))

function _manyplus_negative_entropy(output, inputs)
    _, output_variance = mean_var(output)
    input_variances = map(input -> last(mean_var(input)), inputs)

    # The joint input precision is diag(1 ./ variances) + 11ᵀ / output_variance.
    # The matrix determinant lemma avoids constructing this dense matrix.
    logdet_precision =
        mapreduce(variance -> -log(variance), +, input_variances) +
        log1p(sum(input_variances) / output_variance)

    dimension = length(input_variances)
    one_value = one(logdet_precision)
    two_value = one_value + one_value
    log_two_pi = log(two_value * oftype(logdet_precision, pi))

    return (logdet_precision - dimension * (one_value + log_two_pi)) / two_value
end

function score(
    ::Type{T},
    ::FactorBoundFreeEnergy,
    ::Deterministic,
    node::ManyPlusFactorNode,
    meta,
    stream_postprocessors,
) where {T <: CountingReal}
    inbound_stream(interface) =
        get_stream_of_inbound_messages(interface) |> skip_initial()

    stream = combineLatest(map(inbound_stream, getinterfaces(node)), PushNew())

    mapping =
        messages -> begin
            output = getdata(messages[1])
            inputs = map(getdata, Base.tail(messages))
            return convert(T, _manyplus_negative_entropy(output, inputs))
        end

    scores = stream |> map(T, mapping)
    return postprocess_stream_of_scores(stream_postprocessors, scores)
end
