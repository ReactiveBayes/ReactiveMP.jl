export VariableBoundEntropy

struct VariableBoundEntropy end

function score(::Type{T}, ::VariableBoundEntropy, variable::RandomVariable, scheduler) where T
    mapping = let d = degree(variable)
        (m) -> (d - 1) * convert(InfCountingReal{T}, score(DifferentialEntropy(), m))
    end
    return getmarginal(variable) |> schedule_on(scheduler) |> map(InfCountingReal{T}, mapping)
end