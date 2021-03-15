export score, AverageEnergy, DifferentialEntropy, BetheFreeEnergy
export @average_energy

function score end

struct AverageEnergy end
struct DifferentialEntropy end

struct BetheFreeEnergy end

# Default version is differentiable
# Specialized versions like score(Float64, ...) are not differentiable, but could be faster
score(::BetheFreeEnergy, model)                                           = score(BetheFreeEnergy(), model, AsapScheduler())
score(::BetheFreeEnergy, model, scheduler)                                = score(InfCountingReal, BetheFreeEnergy(), model, scheduler)
score(::Type{T}, ::BetheFreeEnergy, model) where { T <: Real }            = score(T, BetheFreeEnergy(), model, AsapScheduler())
score(::Type{T}, ::BetheFreeEnergy, model, scheduler) where { T <: Real } = score(InfCountingReal{T}, BetheFreeEnergy(), model, scheduler)

function score(::Type{T}, ::BetheFreeEnergy, model, scheduler) where { T <: InfCountingReal }

    node_bound_free_energies     = map((node) -> score(T, FactorBoundFreeEnergy(), node, scheduler), getnodes(model))
    variable_bound_entropies     = map((v) -> score(T, VariableBoundEntropy(), v, scheduler), getrandom(model))
    node_bound_free_energies_sum = collectLatest(T, T, node_bound_free_energies, reduce_with_sum)
    variable_bound_entropies_sum = collectLatest(T, T, variable_bound_entropies, reduce_with_sum)

    point_entropies = Infinity(mapreduce(degree, +, getdata(model), init = 0) + mapreduce(degree, +, getconstant(model), init = 0))

    return combineLatest((node_bound_free_energies_sum, variable_bound_entropies_sum), PushNew()) |> map(eltype(T), d -> float(d[1] + d[2]))
end

## Average energy function helpers

function score(::AverageEnergy, fform, ::Type{ <: Val }, marginals::Tuple{ <: Marginal{ <: NamedTuple{ N } } }, meta) where N
    return score(AverageEnergy(), fform, Val{ N }, map(as_marginal, values(getdata(marginals[1]))), meta)
end

## Differential entropy function helpers

score(::DifferentialEntropy, marginal::Marginal{ <: NamedTuple }) = mapreduce((d) -> score(DifferentialEntropy(), as_marginal(d)), +, getdata(marginal))
score(::DifferentialEntropy, marginal::Marginal)                  = entropy(marginal)

## Average enery macro helper

import .MacroHelpers

macro average_energy(fformtype, lambda)
    @capture(lambda, (args_ where { whereargs__ } = body_) | (args_ = body_)) ||
        error("Error in macro. Lambda body specification is incorrect")

    @capture(args, (inputs__, meta::metatype_) | (inputs__, )) ||
        error("Error in macro. Lambda body arguments speicifcation is incorrect")

    fuppertype = MacroHelpers.upper_type(fformtype)
    whereargs  = whereargs === nothing ? [] : whereargs
    metatype   = metatype === nothing ? :Any : metatype

    inputs = map(inputs) do input
        @capture(input, iname_::itype_) || error("Error in macro. Input $(input) is incorrect")
        return (iname, itype)
    end

    q_names, q_types, q_init_block = rule_macro_parse_fn_args(inputs; specname = :marginals, prefix = :q_, proxy = :Marginal)

    result = quote
        function ReactiveMP.score(
            ::AverageEnergy,
            fform           :: $(fuppertype),
            marginals_names :: $(q_names),
            marginals       :: $(q_types),
            meta            :: $(metatype)
        ) where { $(whereargs...) }
            $(q_init_block...)
            $(body)
        end
    end

    return esc(result)
end
