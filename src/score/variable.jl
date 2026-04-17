export VariableBoundEntropy

struct VariableBoundEntropy end

function score(
    ::Type{T},
    ::VariableBoundEntropy,
    variable::RandomVariable,
    stream_postprocessors,
) where {T <: CountingReal}
    mapping = let d = degree(variable)
        (marginal) -> begin
            # The entropy of point masses is not finite
            # In this case we treat them as clamped variables, such that we should multiply
            # their influence on `d` instead of `d - 1`
            scaling = !ispointmass(marginal) ? (d - 1) : d
            entropy = convert(T, score(DifferentialEntropy(), marginal))
            return scaling * entropy
        end
    end
    stream_of_scores =
        get_stream_of_marginals(variable) |> skip_initial() |> map(T, mapping)
    stream_of_scores = postprocess_stream_of_scores(
        stream_postprocessors, stream_of_scores
    )
    return stream_of_scores
end
