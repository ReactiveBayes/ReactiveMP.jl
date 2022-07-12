using Random

# We define a rule for `DeltaFn{f}` where `f` is a callable reference to our function and can be called as `f(1, 2, 3)` blabla
# `m_ins` is a tuple of input messages
# `meta` handles reference to our meta object
# `N` can be used for dispatch and can handle special cases, e.g `m_ins::NTuple{1, NormalMeanPrecision}` means that `DeltaFn` has only 1 input

@rule DeltaFn{f}(:out, Marginalisation) (m_ins::NTuple{N, Any}, meta::SamplingApproximation) where {f, N} =
    begin
        message_samples = rand(meta.rng, m_in, meta.nsamples)
        return SampleList(map(x -> f(x...), message_samples))
    end

@rule DeltaFn{f}(:out, Marginalisation) (m_out::Any, m_ins::NTuple{1, Any}, meta::LinearApproximation) where {f} = begin
    m1, v1 = mean(m_ins[1]), var(m_ins[1])
    (a, b) = localLinearization(f, m1)
    m = a * m1 + b
    V = a * v1 * a'
    return NormalMeanVariance(m, V)
end

@rule DeltaFn{f}(:out, Marginalisation) (m_out::Any, m_in::Any, meta::LinearApproximation) where {f} = begin
    mean, var = mean(m_ins[1]), var(m_ins[1])
    (a, b) = localLinearization(g, mean)
    m = a * mean + b
    V = a * var * a'
    return NormalMeanVariance(m, V)
end

# function ruleSPCVIIn1Factor(node_id::Symbol,
#     msg_out::Message{<:FactorFunction, <:VariateType},
#     msg_in::Message{<:FactorNode, <:VariateType})
#     thenode = currentGraph().nodes[node_id]

#     η = deepcopy(naturalParams(msg_in.dist))
#     if thenode.online_inference == false
#         λ_init = deepcopy(η)
#     else
#         λ_init = deepcopy(naturalParams(thenode.q[1]))
#     end

#     logp_nc(z) = (thenode.dataset_size / thenode.batch_size) * logPdf(msg_out.dist, thenode.g(z))
#     λ = renderCVI(logp_nc, thenode.num_iterations, thenode.opt, λ_init, msg_in, thenode.convergence_optimizer)

#     λ_message = λ .- η
#     # Implement proper message check for all the distributions later on.
#     thenode.q = [standardDist(msg_in.dist, λ)]
#     if thenode.online_inference == false
#         thenode.q_memory = deepcopy(thenode.q)
#     end
#     return standardMessage(msg_in.dist, λ_message)
# end

function ruleSPCVIOutNFactorNode(node_id::Symbol,
    msg_out::Nothing,
    msg_in::Message)
    thenode = currentGraph().nodes[node_id]

    sampl = thenode.g(sample(msg_in.dist))
    if length(sampl) == 1
        variate = Univariate
    else
        variate = Multivariate
    end
    return Message(variate, SetSampleList, node_id = node_id)
end

@rule DeltaFn{f}(:out, Marginalisation) (m_out::Any, m_ins::NTuple{N, Any}, meta::CVIApproximation) where {f, N} = begin
    return NormalMeanVariance(0, 1)
end

# @rule DeltaFn{f}(:out, Marginalisation) (m_ins::NTuple{N, Any}, meta::LinearApproximationKnownInverse) where {f, N} =
#     begin
#         return NormalMeanPrecision(f(mean.(m_ins)...), 1.0)
#     end

# @rule DeltaFn{f}(:out, Marginalisation) (m_ins::NTuple{N, Any}, meta::LinearApproximationUnknownInverse) where {f, N} =
#     begin
#         return NormalMeanPrecision(f(mean.(m_ins)...), 1.0)
#     end
