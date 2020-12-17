export VariableBoundEntropy

struct VariableBoundEntropy end

function score(::VariableBoundEntropy, variable::RandomVariable, scheduler)
    mapping = let d = degree(variable)
        (m) -> convert(InfCountingReal, (d - 1) * score(DifferentialEntropy(), m))
    end
    return getmarginal(variable) |> schedule_on(scheduler) |> map(InfCountingReal, mapping)
end