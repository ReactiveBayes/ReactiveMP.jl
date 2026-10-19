export VariableBoundEntropy

"""
    VariableBoundEntropy()

Selects a random variable's contribution to the Bethe free energy in [`score`](@ref): the entropy
of its marginal weighted by `d - 1`, `d` being its [`ReactiveMP.degree`](@ref), or by `d` for a
point-mass marginal, whose entropy is not finite.

The stream reports [`ReactiveMP.BeforeVariableBoundEntropyEvent`](@ref) and
[`ReactiveMP.AfterVariableBoundEntropyEvent`](@ref) to the callbacks the variable was activated
with, those of its `prod_context_for_marginal_computation` (see
[`RandomVariableActivationOptions`](@ref)), read when the stream is built: activate the variable
first.
"""
struct VariableBoundEntropy end

function score(
        ::Type{T},
        ::VariableBoundEntropy,
        variable::RandomVariable,
        stream_postprocessors,
    ) where {T <: CountingReal}
    mapping = variable_entropy_term(T, variable, variable.callbacks)
    stream_of_scores =
        get_stream_of_marginals(variable) |> skip_initial() |> map(T, mapping)
    stream_of_scores = postprocess_stream_of_scores(
        stream_postprocessors, stream_of_scores
    )
    return stream_of_scores
end

# The function from a variable's marginal to its term, between the free-energy events. The
# callbacks are read once, when the stream is built, and this barrier makes them concrete in it.
function variable_entropy_term(::Type{T}, variable, callbacks) where {T}
    d = degree(variable)
    return (marginal) -> begin
        span_id = generate_span_id(callbacks)
        @invoke_callback(callbacks, BeforeVariableBoundEntropyEvent(variable, marginal, span_id))
        # The entropy of point masses is not finite
        # In this case we treat them as clamped variables, such that we should multiply
        # their influence on `d` instead of `d - 1`
        scaling = !ispointmass(marginal) ? (d - 1) : d
        entropy = convert(T, score(DifferentialEntropy(), marginal))
        result = scaling * entropy
        @invoke_callback(callbacks, AfterVariableBoundEntropyEvent(variable, marginal, entropy, scaling, result, span_id))
        return result
    end
end
