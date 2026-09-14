# The numerical Delta rules need a function, not a collection of observables.
struct CompiledDeltaContext{F, P} <: AbstractFactorNode
    fn::F
    proxy::P
end
functionalform(::CompiledDeltaContext{F}) where {F} = DeltaFn{F}
sdtype(::CompiledDeltaContext) = Deterministic()
nodefunction(node::CompiledDeltaContext, ::DeltaMeta, ::Val{:out}) = node.proxy
nodefunction(::CompiledDeltaContext, meta::DeltaMeta, ::Val{:in}) = getinverse(meta)
nodefunction(::CompiledDeltaContext, meta::DeltaMeta, ::Val{:in}, k::Integer) = getinverse(meta, k)

struct CompiledDeltaProxy{F, A, S}
    fn::F
    arguments::A # positive: random input; negative: observed/static input
    statics::S
end
function (proxy::CompiledDeltaProxy)(randoms...)
    args = map(i -> i > 0 ? randoms[i] : mean(getdata(proxy.statics[-i])), proxy.arguments)
    return proxy.fn(args...)
end

struct CompiledDeltaKernel{F, A, M}
    fn::F
    arguments::A
    mapping::M
end
function (kernel::CompiledDeltaKernel)(inputs)
    dependencies, statics = inputs
    if any(input -> ismissing(getdata(input)), statics)
        return kernel.mapping isa MessageMapping ? Message(missing, false, false) : Marginal(missing, false, false)
    end
    proxy = CompiledDeltaProxy(kernel.fn, kernel.arguments, statics)
    node = CompiledDeltaContext(kernel.fn, proxy)
    return compiled_delta_apply(kernel.mapping, node, dependencies)
end
function compiled_delta_apply(mapping::MessageMapping, node, inputs)
    rebound = MessageMapping(message_mapping_fform(mapping), mapping.vtag, mapping.vconstraint,
        mapping.msgs_names, mapping.marginals_names, mapping.meta, mapping.annotations,
        node, mapping.rulefallback, mapping.callbacks)
    return rebound(inputs[1], inputs[2])
end
function compiled_delta_apply(mapping::MarginalMapping, node, inputs)
    rebound = MarginalMapping(marginal_mapping_fform(mapping), mapping.vtag,
        mapping.msgs_names, mapping.marginals_names, mapping.meta, node)
    return compute_marginal(rebound, inputs[1], inputs[2])
end
# CVI carries mutable optimization/RNG state; keep it serial even when its
# default RNG is task-local. Deterministic, immutable methods can run in parallel.
compiled_parallel_safe(kernel::CompiledDeltaKernel) =
    !(getmethod(kernel.mapping.meta) isa CVI) && multicore_readonly(kernel)
