export score, AverageEnergy, DifferentialEntropy, BetheFreeEnergy
export BetheFreeEnergyCheckNaNs, BetheFreeEnergyCheckInfs, BetheFreeEnergyDefaultChecks
export @average_energy

abstract type AbstractScoreObjective end

function score end

# Default version is differentiable
# Specialized versions like score(Float64, ...) are not differentiable, but could be faster
score(objective::AbstractScoreObjective, model)            = score(objective, model, AsapScheduler())
score(objective::AbstractScoreObjective, model, scheduler) = score(InfCountingReal, objective, model, scheduler)

score(::Type{T}, objective::AbstractScoreObjective, model) where {T <: Real}            = score(T, objective, model, AsapScheduler())
score(::Type{T}, objective::AbstractScoreObjective, model) where {T <: InfCountingReal} = score(T, objective, model, AsapScheduler())

score(::Type{T}, objective::AbstractScoreObjective, model, scheduler) where {T <: Real}            = score(InfCountingReal{T}, objective, model, scheduler)
score(::Type{T}, objective::AbstractScoreObjective, model, scheduler) where {T <: InfCountingReal} = score(T, objective, model, scheduler)

# Bethe Free Energy objective

## Various check/report structures

"""
    apply_diagnostic_check(check, context, stream)

This function applies a `check` to the `stream`. Accepts optional context object for custom error messages.
"""
function apply_diagnostic_check end

"""
    BetheFreeEnergyCheckNaNs

If enabled checks that both variable and factor bound score functions in Bethe Free Energy computation do not return `NaN`s. 
Throws an error if finds `NaN`. 

See also: [`BetheFreeEnergyCheckInfs`](@ref)
"""
struct BetheFreeEnergyCheckNaNs end

function apply_diagnostic_check(::BetheFreeEnergyCheckNaNs, node::AbstractFactorNode, stream)
    return stream |> error_if(value_isnan, (_) -> """
        Failed to compute node bound free energy component. The result is `NaN`. 
        Use `diagnostic_checks` field in `BetheFreeEnergy` constructor or `free_energy_diagnostics` keyword argument in the `inference` function to suppress this error.
        $(node)
    """)
end

function apply_diagnostic_check(::BetheFreeEnergyCheckNaNs, variable::AbstractVariable, stream)
    return stream |> error_if(value_isnan, (_) -> """
        Failed to compute variable bound free energy component for `$(indexed_name(variable))` variable. The result is `NaN`. 
        Use `diagnostic_checks` field in `BetheFreeEnergy` constructor or `free_energy_diagnostics` keyword argument in the `inference` function to suppress this error.
    """)
end

"""
    BetheFreeEnergyCheckInfs

If enabled checks that both variable and factor bound score functions in Bethe Free Energy computation do not return `Inf`s. 
Throws an error if finds `Inf`. 

See also: [`BetheFreeEnergyCheckNaNs`](@ref)
"""
struct BetheFreeEnergyCheckInfs end

function apply_diagnostic_check(::BetheFreeEnergyCheckInfs, node::AbstractFactorNode, stream)
    return stream |> error_if(value_isinf, (_) -> """
        Failed to compute node bound free energy component. The result is `Inf`. 
        Use `diagnostic_checks` field in `BetheFreeEnergy` constructor or `free_energy_diagnostics` keyword argument in the `inference` function to suppress this error.
        $(node)
    """)
end

function apply_diagnostic_check(::BetheFreeEnergyCheckInfs, variable::AbstractVariable, stream)
    return stream |> error_if(value_isnan, (_) -> """
        Failed to compute variable bound free energy component for `$(indexed_name(variable))` variable. The result is `Inf`. 
        Use `diagnostic_checks` field in `BetheFreeEnergy` constructor or `free_energy_diagnostics` keyword argument in the `inference` function to suppress this error.
    """)
end

apply_diagnostic_check(::Nothing, something, stream)     = stream
apply_diagnostic_check(checks::Tuple, something, stream) = foldl((folded, check) -> apply_diagnostic_check(check, something, folded), checks; init = stream)

##

struct AverageEnergy end
struct DifferentialEntropy end

"""
    BetheFreeEnergy(marginal_skip_strategy, diagnostic_checks)

Creates Bethe Free Energy values stream when passed to the `score` function. 
"""
struct BetheFreeEnergy{S, C} <: AbstractScoreObjective
    marginal_skip_strategy::S
    diagnostic_checks::C
end

const BetheFreeEnergyDefaultMarginalSkipStrategy = SkipInitial()
const BetheFreeEnergyDefaultChecks               = (BetheFreeEnergyCheckNaNs(), BetheFreeEnergyCheckInfs())

BetheFreeEnergy() = BetheFreeEnergy(BetheFreeEnergyDefaultMarginalSkipStrategy, BetheFreeEnergyDefaultChecks)

marginal_skip_strategy(objective::BetheFreeEnergy) = objective.marginal_skip_strategy

apply_diagnostic_check(objective::BetheFreeEnergy, something, stream) =
    apply_diagnostic_check(objective.diagnostic_checks, something, stream)

function score(::Type{T}, objective::BetheFreeEnergy, model, scheduler) where {T <: InfCountingReal}
    stochastic_variables = filter(r -> !is_point_mass_form_constraint(marginal_form_constraint(r)), getrandom(model))
    point_mass_estimates = filter(r -> is_point_mass_form_constraint(marginal_form_constraint(r)), getrandom(model))

    node_bound_free_energies     = map((node) -> score(T, objective, FactorBoundFreeEnergy(), node, scheduler), getnodes(model))
    variable_bound_entropies     = map((v) -> score(T, objective, VariableBoundEntropy(), v, scheduler), stochastic_variables)
    node_bound_free_energies_sum = collectLatest(T, T, node_bound_free_energies, reduce_with_sum)
    variable_bound_entropies_sum = collectLatest(T, T, variable_bound_entropies, reduce_with_sum)

    data_point_entropies_n     = mapreduce(degree, +, getdata(model), init = 0)
    constant_point_entropies_n = mapreduce(degree, +, getconstant(model), init = 0)
    form_point_entropies_n     = mapreduce(degree, +, point_mass_estimates, init = 0)

    point_entropies =
        InfCountingReal(eltype(T), data_point_entropies_n + constant_point_entropies_n + form_point_entropies_n)

    return combineLatest((node_bound_free_energies_sum, variable_bound_entropies_sum), PushNew()) |>
           map(eltype(T), d -> float(d[1] + d[2] - point_entropies))
end

## Average energy function helpers

function score(::AverageEnergy, fform, ::Type{<:Val}, marginals::Tuple{<:Marginal{<:NamedTuple{N}}}, meta) where {N}
    joint = marginals[1]

    transform = let is_joint_clamped = is_clamped(joint), is_joint_initial = is_initial(joint)
        (data) -> Marginal(data, is_joint_clamped, is_joint_initial)
    end

    return score(AverageEnergy(), fform, Val{N}, map(transform, values(getdata(joint))), meta)
end

## Differential entropy function helpers

score(::DifferentialEntropy, marginal::Marginal) = entropy(marginal)

function score(::DifferentialEntropy, marginal::Marginal{<:NamedTuple})
    compute_score = let is_marginal_clamped = is_clamped(marginal), is_marginal_initial = is_initial(marginal)
        (data) -> score(DifferentialEntropy(), Marginal(data, is_marginal_clamped, is_marginal_initial))
    end

    return mapreduce(compute_score, +, values(getdata(marginal)))
end

## Average enery macro helper

import .MacroHelpers

macro average_energy(fformtype, lambda)
    @capture(lambda, (args_ where {whereargs__} = body_) | (args_ = body_)) ||
        error("Error in macro. Lambda body specification is incorrect")

    @capture(args, (inputs__, meta::metatype_) | (inputs__,)) ||
        error("Error in macro. Lambda body arguments speicifcation is incorrect")

    fuppertype = MacroHelpers.upper_type(fformtype)
    whereargs  = whereargs === nothing ? [] : whereargs
    metatype   = metatype === nothing ? :Any : metatype

    inputs = map(inputs) do input
        @capture(input, iname_::itype_) || error("Error in macro. Input $(input) is incorrect")
        return (iname, itype)
    end

    q_names, q_types, q_init_block =
        rule_macro_parse_fn_args(inputs; specname = :marginals, prefix = :q_, proxy = :Marginal)

    result = quote
        function ReactiveMP.score(
            ::AverageEnergy,
            fform::$(fuppertype),
            marginals_names::$(q_names),
            marginals::$(q_types),
            meta::$(metatype)
        ) where {$(whereargs...)}
            $(q_init_block...)
            $(body)
        end
    end

    return esc(result)
end
