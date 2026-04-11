export score, AverageEnergy, DifferentialEntropy
export @average_energy

"""
    score(type, args...)

Central dispatch point for computing local free-energy contributions during
inference. Called internally by the engine whenever a marginal changes.

The first argument is a tag type that selects the kind of score:

- `score(AverageEnergy(), fform, marginal_names, marginals, meta)` — computes
  `⟨-log f⟩_q` for a factor node `fform` under the local joint marginal `q`.
- `score(DifferentialEntropy(), marginal)` — computes the Shannon entropy `H[q]`
  of a marginal.

Use [`@average_energy`](@ref) to define the `AverageEnergy` score for a custom
factor node.
"""
function score end

##

"""
    AverageEnergy

Dispatch tag for computing the average energy `⟨-log f⟩_q` of a factor node
under the local joint marginal. Used as the first argument to [`score`](@ref).

Define the average energy for a custom node with the [`@average_energy`](@ref) macro.
"""
struct AverageEnergy end

"""
    DifferentialEntropy

Dispatch tag for computing the Shannon (differential) entropy `H[q] = -∫ q log q`
of a marginal distribution. Used as the first argument to [`score`](@ref).
"""
struct DifferentialEntropy end

struct KLDivergence end

## We have a special case of marginals, that are represented as NamedTuple, 
## in this case we need to decompose it into separate marginals and inject them into the score function recursively
## This function is used to extract the index of the named tuple-based marginal (if it exists)
scan_marginals_for_named_tuple(::Val{Index}, current::Marginal{<:NamedTuple{N}}, rest::Tuple) where {Index, N} = (
    Val{Index}(), Val{N}(), current
)
scan_marginals_for_named_tuple(::Val{Index}, current::Marginal{<:NamedTuple{N}}, rest::Tuple{}) where {Index, N} = (
    Val{Index}(), Val{N}(), current
)
scan_marginals_for_named_tuple(::Val{Index}, current::Marginal, rest::Tuple) where {Index} = scan_marginals_for_named_tuple(
    Val{Index + 1}(), rest[1], rest[2:end]
)
scan_marginals_for_named_tuple(::Val{Index}, current::Marginal, rest::Tuple{}) where {Index} = (
    nothing, nothing, nothing
)

function score(
    ::AverageEnergy, fform, ::Val{Names}, marginals::Tuple, meta
) where {Names}
    # Generic method for the Average Score computation tries to scan the marginals for:
    # NamedTuple-based marginals
    # - If found, it decomposes the joint marginal into separate marginals and injects them into the score function recursively
    valIndex, valN, joint = scan_marginals_for_named_tuple(
        Val(1), marginals[1], marginals[2:end]
    )

    if !isnothing(valIndex)
        transform =
            let is_joint_clamped = is_clamped(joint),
                is_joint_initial = is_initial(joint)

                (data) -> Marginal(data, is_joint_clamped, is_joint_initial)
            end

        mod_marginals = TupleTools.insertat(
            marginals, unval(valIndex), map(transform, values(getdata(joint)))
        )
        mod_names = TupleTools.insertat(Names, unval(valIndex), unval(valN))

        return score(
            AverageEnergy(), fform, Val{mod_names}(), mod_marginals, meta
        )
    end

    # - If not found, the method throws an error suggesting to use 
    # the `@average_energy` macro to define the method for the provided marginals
    error_names = map(n -> string(:q_, n), Names)
    error_types = map(m -> typeofdata(m), marginals)
    error_suggestion_args = join(
        map((z) -> string(z[1], "::", z[2]), zip(error_names, error_types)),
        ", ",
    )

    error(
        """ 
  Cannot compute Average Energy for the $(fform) node, the method does not exist for the provided marginals.
  Use the `@average_energy` macro to define the method for the provided marginals, e.g.

      @average_energy $(fform) ($error_suggestion_args, $(ifelse(isnothing(meta), "", "meta::$(typeof(meta))"))) begin
          # ...
      end

  """,
    )
end

## Differential entropy function helpers

score(::DifferentialEntropy, marginal::Marginal) = entropy(marginal)

function score(::DifferentialEntropy, marginal::Marginal{<:NamedTuple})
    compute_score =
        let is_marginal_clamped = is_clamped(marginal),
            is_marginal_initial = is_initial(marginal)

            (data) -> score(
                DifferentialEntropy(),
                Marginal(data, is_marginal_clamped, is_marginal_initial),
            )
        end

    return mapreduce(compute_score, +, values(getdata(marginal)))
end

## Kl KlDivergence

score(::KLDivergence, marginal::Marginal, p::Distribution) = Distributions.kldivergence(
    getdata(marginal), p
)

## Average energy macro helper

import .MacroHelpers

"""
    @average_energy NodeType (q_x::DistType, q_y::DistType, ...) begin
        # return -⟨log f(x, y, ...)⟩_{q(x)q(y)...}
    end

Define the average energy `⟨-log f⟩_q` for a custom factor node. Generates a
`score(::AverageEnergy, ...)` method dispatched on the node type and the types
of the marginals.

Marginal arguments must be named with a `q_` prefix matching the interface names
declared in the corresponding [`@node`](@ref) definition. An optional `meta`
argument of a specific type may be included as the last argument.

# Example

```julia
@node MyNode Stochastic [out, x, y]

@average_energy MyNode (q_out::Any, q_x::NormalMeanVariance, q_y::Gamma) begin
    mx, vx = mean_var(q_x)
    my     = mean(q_y)
    return 0.5 * log(2π) + 0.5 * (vx + mx^2) * my
end
```
"""
macro average_energy(fformtype, lambda)
    @capture(lambda, (args_ where {whereargs__} = body_) | (args_ = body_)) ||
        error("Error in macro. Lambda body specification is incorrect")

    @capture(args, (inputs__, meta::metatype_) | (inputs__,)) || error(
        "Error in macro. Lambda body arguments speicifcation is incorrect"
    )

    fuppertype = MacroHelpers.upper_type(fformtype)
    fbottomtype = MacroHelpers.bottom_type(fformtype)
    whereargs = whereargs === nothing ? [] : whereargs
    metatype = metatype === nothing ? :Nothing : metatype

    inputs = map(inputs) do input
        @capture(input, iname_::itype_) ||
            error("Error in macro. Input $(input) is incorrect")
        return (iname, itype)
    end

    rule_macro_check_fn_args(
        inputs; allowed_inputs = (:meta,), allowed_prefixes = (:q_,)
    )

    q_names, q_types, q_init_block = rule_macro_parse_fn_args(
        inputs; specname = :marginals, prefix = :q_, proxy = :Marginal
    )

    # Some `@rules` are more complex in terms of functional form specification, e.g. `NormalMixture{N}`
    if fbottomtype isa Symbol
        # Not all nodes are defined with the `@node` macro, so we need to check if the node is defined with the `@node` macro
        # `nodesymbol_to_nodefform` may return `nothing` for such nodes, in this case we skip the interface check
        nodefform_from_symbol = ReactiveMP.nodesymbol_to_nodefform(
            Val(fbottomtype)
        )
        if !isnothing(nodefform_from_symbol)
            ifaces = ReactiveMP.interfaces(nodefform_from_symbol)
            MacroHelpers.check_rule_interfaces(
                "@average_energy",
                fformtype,
                lambda,
                ifaces,
                nothing,
                nothing,
                q_names;
                mod = __module__,
            )
        end
    end

    result = quote
        function ReactiveMP.score(
            ::AverageEnergy,
            fform::$(fuppertype),
            marginals_names::$(q_names),
            marginals::$(q_types),
            meta::$(metatype),
        ) where {$(whereargs...)}
            $(q_init_block...)
            $(body)
        end
    end

    return esc(result)
end
