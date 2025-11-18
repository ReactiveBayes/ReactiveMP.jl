export score, AverageEnergy, DifferentialEntropy
export @average_energy

function score end

##

struct AverageEnergy end

struct DifferentialEntropy end

struct KLDivergence end

## We have a special case of marginals, that are represented as NamedTuple, 
## in this case we need to decompose it into separate marginals and inject them into the score function recursively
## This function is used to extract the index of the named tuple-based marginal (if it exists)
scan_marginals_for_named_tuple(::Val{Index}, current::Marginal{<:NamedTuple{N}}, rest::Tuple) where {Index, N} = (Val{Index}(), Val{N}(), current)
scan_marginals_for_named_tuple(::Val{Index}, current::Marginal{<:NamedTuple{N}}, rest::Tuple{}) where {Index, N} = (Val{Index}(), Val{N}(), current)
scan_marginals_for_named_tuple(::Val{Index}, current::Marginal, rest::Tuple) where {Index} = scan_marginals_for_named_tuple(Val{Index + 1}(), rest[1], rest[2:end])
scan_marginals_for_named_tuple(::Val{Index}, current::Marginal, rest::Tuple{}) where {Index} = (nothing, nothing, nothing)

function score(::AverageEnergy, fform, ::Val{Names}, marginals::Tuple, meta) where {Names}
    # Generic method for the Average Score computation tries to scan the marginals for:
    # NamedTuple-based marginals
    # - If found, it decomposes the joint marginal into separate marginals and injects them into the score function recursively
    valIndex, valN, joint = scan_marginals_for_named_tuple(Val(1), marginals[1], marginals[2:end])

    if !isnothing(valIndex)
        transform = let is_joint_clamped = is_clamped(joint), is_joint_initial = is_initial(joint)
            (data) -> Marginal(data, is_joint_clamped, is_joint_initial, nothing)
        end

        mod_marginals = TupleTools.insertat(marginals, unval(valIndex), map(transform, values(getdata(joint))))
        mod_names = TupleTools.insertat(Names, unval(valIndex), unval(valN))

        return score(AverageEnergy(), fform, Val{mod_names}(), mod_marginals, meta)
    end

    # - If not found, the method throws an error suggesting to use 
    # the `@average_energy` macro to define the method for the provided marginals
    error_names = map(n -> string(:q_, n), Names)
    error_types = map(m -> typeofdata(m), marginals)
    error_suggestion_args = join(map((z) -> string(z[1], "::", z[2]), zip(error_names, error_types)), ", ")

    error(""" 
    Cannot compute Average Energy for the $(fform) node, the method does not exist for the provided marginals.
    Use the `@average_energy` macro to define the method for the provided marginals, e.g.

        @average_energy $(fform) ($error_suggestion_args, $(ifelse(isnothing(meta), "", "meta::$(typeof(meta))"))) begin
            # ...
        end

    """)
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
    fbottomtype = MacroHelpers.bottom_type(fformtype)
    whereargs = whereargs === nothing ? [] : whereargs
    metatype = metatype === nothing ? :Nothing : metatype

    inputs = map(inputs) do input
        @capture(input, iname_::itype_) || error("Error in macro. Input $(input) is incorrect")
        return (iname, itype)
    end

    rule_macro_check_fn_args(inputs; allowed_inputs = (:meta,), allowed_prefixes = (:q_,))

    q_names, q_types, q_init_block = rule_macro_parse_fn_args(inputs; specname = :marginals, prefix = :q_, proxy = :Marginal)

    # Some `@rules` are more complex in terms of functional form specification, e.g. `NormalMixture{N}`
    if fbottomtype isa Symbol
        # Not all nodes are defined with the `@node` macro, so we need to check if the node is defined with the `@node` macro
        # `nodesymbol_to_nodefform` may return `nothing` for such nodes, in this case we skip the interface check
        nodefform_from_symbol = ReactiveMP.nodesymbol_to_nodefform(Val(fbottomtype))
        if !isnothing(nodefform_from_symbol)
            ifaces = ReactiveMP.interfaces(nodefform_from_symbol)
            MacroHelpers.check_rule_interfaces("@average_energy", fformtype, lambda, ifaces, nothing, nothing, q_names; mod = __module__)
        end
    end

    result = quote
        function ReactiveMP.score(::AverageEnergy, fform::$(fuppertype), marginals_names::$(q_names), marginals::$(q_types), meta::$(metatype)) where {$(whereargs...)}
            $(q_init_block...)
            $(body)
        end
    end

    return esc(result)
end
