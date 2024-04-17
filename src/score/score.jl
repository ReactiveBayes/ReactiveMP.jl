export score, AverageEnergy, DifferentialEntropy
export @average_energy

function score end

##

struct AverageEnergy end

struct DifferentialEntropy end

struct KLDivergence end

## Average energy function helpers

function score(::AverageEnergy, fform, ::Val, marginals::Tuple{<:Marginal{<:NamedTuple{N}}}, meta) where {N}
    joint = marginals[1]

    transform = let is_joint_clamped = is_clamped(joint), is_joint_initial = is_initial(joint)
        (data) -> Marginal(data, is_joint_clamped, is_joint_initial, nothing)
    end

    return score(AverageEnergy(), fform, Val{N}(), map(transform, values(getdata(joint))), meta)
end

## Differential entropy function helpers

score(::DifferentialEntropy, marginal::Marginal) = entropy(marginal)

function score(::DifferentialEntropy, marginal::Marginal{<:NamedTuple})
    compute_score = let is_marginal_clamped = is_clamped(marginal), is_marginal_initial = is_initial(marginal)
        (data) -> score(DifferentialEntropy(), Marginal(data, is_marginal_clamped, is_marginal_initial, nothing))
    end

    return mapreduce(compute_score, +, values(getdata(marginal)))
end

## Kl KlDivergence

score(::KLDivergence, marginal::Marginal, p::Distribution) = Distributions.kldivergence(getdata(marginal), p)

## Average enery macro helper

import .MacroHelpers

macro average_energy(fformtype, lambda)
    @capture(lambda, (args_ where {whereargs__} = body_) | (args_ = body_)) || error("Error in macro. Lambda body specification is incorrect")

    @capture(args, (inputs__, meta::metatype_) | (inputs__,)) || error("Error in macro. Lambda body arguments speicifcation is incorrect")

    fuppertype = MacroHelpers.upper_type(fformtype)
    whereargs  = whereargs === nothing ? [] : whereargs
    metatype   = metatype === nothing ? :Nothing : metatype

    inputs = map(inputs) do input
        @capture(input, iname_::itype_) || error("Error in macro. Input $(input) is incorrect")
        return (iname, itype)
    end

    rule_macro_check_fn_args(inputs; allowed_inputs = (:meta,), allowed_prefixes = (:q_,))

    q_names, q_types, q_init_block = rule_macro_parse_fn_args(inputs; specname = :marginals, prefix = :q_, proxy = :Marginal)

    result = quote
        function ReactiveMP.score(::AverageEnergy, fform::$(fuppertype), marginals_names::$(q_names), marginals::$(q_types), meta::$(metatype)) where {$(whereargs...)}
            $(q_init_block...)
            $(body)
        end
    end

    return esc(result)
end
