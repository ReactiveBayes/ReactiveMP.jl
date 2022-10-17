using Parameters

export DeltaUnscented, UT

const default_alpha = 1e-3 # Default value for the spread parameter
const default_beta = 2.0
const default_kappa = 0.0

@with_kw struct DeltaUnscented{T}
    inverse::T = nothing
    alpha::Real = default_alpha
    beta::Real = default_beta
    kappa::Real = default_kappa
end

const UT = DeltaUnscented

deltafn_rule_layout(::DeltaFnNode, ::DeltaUnscented{Nothing}) =
    DeltaUknownInverseApproximationDeltaFnRuleLayout()

deltafn_rule_layout(::DeltaFnNode, ::DeltaUnscented{F}) where {F <: Function}               = DeltaKnownInverseApproximationDeltaFnRuleLayout()
deltafn_rule_layout(::DeltaFnNode, ::DeltaUnscented{F}) where {N, F <: NTuple{N, Function}} = DeltaKnownInverseApproximationDeltaFnRuleLayout()
