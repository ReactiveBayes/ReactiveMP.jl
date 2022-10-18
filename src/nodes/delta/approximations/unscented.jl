export DeltaUnscented, UT

const default_alpha = 1e-3 # Default value for the spread parameter
const default_beta = 2.0
const default_kappa = 0.0

Base.@kwdef struct DeltaUnscented{T, A, B, K}
    inverse::T = nothing
    alpha::A = default_alpha
    beta::B = default_beta
    kappa::K = default_kappa
end

const UT = DeltaUnscented

deltafn_rule_layout(::DeltaFnNode, ::DeltaUnscented{Nothing}) =
    DeltaUknownInverseApproximationDeltaFnRuleLayout()

deltafn_rule_layout(::DeltaFnNode, ::DeltaUnscented{F}) where {F <: Function}               = DeltaKnownInverseApproximationDeltaFnRuleLayout()
deltafn_rule_layout(::DeltaFnNode, ::DeltaUnscented{F}) where {N, F <: NTuple{N, Function}} = DeltaKnownInverseApproximationDeltaFnRuleLayout()
