export score, AverageEnergy, DifferentialEntropy, BetheFreeEnergy
export @average_energy

function score end

struct AverageEnergy end
struct DifferentialEntropy end

struct BetheFreeEnergy end

function score(::BetheFreeEnergy, model, scheduler)
    return score(Float64, BetheFreeEnergy(), model, scheduler)
end

function score(::Type{T}, ::BetheFreeEnergy, model, scheduler) where T

    node_energies = map(filter(isstochastic, getnodes(model))) do node
        marginal_names   = Val{ tuple(clusternames(node)...) } 
        marginals_stream = combineLatest(map(cluster -> getmarginal!(node, cluster), clusters(node)), PushEach())
        return marginals_stream |> schedule_on(scheduler) |> map(InfCountingReal{T}, (marginals) -> begin 
            average_energy   = InfCountingReal(score(AverageEnergy(), functionalform(node), marginal_names, marginals, metadata(node)))
            clusters_entropy = mapreduce(marginal -> score(DifferentialEntropy(), marginal), +, marginals)
            return average_energy - clusters_entropy
        end)
    end

    differential_entropies = map(getrandom(model)) do random 
        d       = degree(random)
        mapping = (m) -> (d - 1) * InfCountingReal(score(DifferentialEntropy(), m))
        return getmarginal(random) |> schedule_on(scheduler) |> map(InfCountingReal{T}, mapping)
    end

    energies_sum     = collectLatest(InfCountingReal{T}, node_energies, InfCountingReal{T}, reduce_with_sum) 
    entropies_sum    = collectLatest(InfCountingReal{T}, differential_entropies, InfCountingReal{T}, reduce_with_sum) 
    diracs_entropies = Infinity(length(getdata(model)) + length(getconstant(model)))

    return combineLatest((energies_sum, entropies_sum), PushNew()) |> map(T, d -> convert(T, d[1] + d[2] - diracs_entropies))
end

## Average energy function helpers

function score(::AverageEnergy, fform, ::Type{ Val{ N } }, marginals::Tuple{ <: Marginal{ <: Tuple } }, meta) where N
    return score(AverageEnergy(), fform, split_underscored_symbol(Val{ N[1] }), map(as_marginal, getdata(marginals[1])), meta)
end

## Differential entropy function helpers

score(::DifferentialEntropy, marginal::Marginal{ <: Tuple }) = mapreduce((d) -> score(DifferentialEntropy(), as_marginal(d)), +, getdata(marginal))
score(::DifferentialEntropy, marginal::Marginal)             = entropy(marginal)

## Average enery macro helper

macro average_energy(fformtype, marginals, meta, fn)
    q_names, q_types, q_init_block, q_where_Ts = __extract_fn_args_macro_rule(marginals; specname = :marginals, prefix = :q_, proxytype = :Marginal)
    
    result = quote
        function ReactiveMP.score(
            ::AverageEnergy,
            fform           :: $(__extract_fformtype_macro_rule(fformtype)),
            marginals_names :: $(q_names),
            marginals       :: $(q_types),
            meta            :: $(__extract_meta_macro_rule(meta))
        ) where { $(q_where_Ts...) }
            $(q_init_block...)
            $(fn)
        end
    end
    
    return esc(result)
end