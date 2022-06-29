export VariableBoundEntropy

struct VariableBoundEntropy end

function score(
    ::Type{T},
    objective::BetheFreeEnergy,
    ::VariableBoundEntropy,
    variable::RandomVariable,
    scheduler
) where {T <: InfCountingReal}
    mapping = let d = degree(variable)
        (marginal) -> convert(T, (d - 1) * score(DifferentialEntropy(), marginal))
    end
    stream = getmarginal(variable, marginal_skip_strategy(objective)) |> schedule_on(scheduler)
    return apply_diagnostic_check(objective, variable, stream |> map(T, mapping))
end
