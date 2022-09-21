using Parameters

export DeltaExtended, ET

@with_kw struct DeltaExtended{T}
    inverse::T = nothing
end

const ET = DeltaExtended

deltafn_rule_layout(::DeltaFnNode, ::DeltaExtended{Nothing}) =
    DeltaUEUknownInverseApproximationDeltaFnRuleLayout()

deltafn_rule_layout(::DeltaFnNode, ::DeltaExtended{F}) where {F <: Function}               = DeltaUEKnownInverseApproximationDeltaFnRuleLayout()
deltafn_rule_layout(::DeltaFnNode, ::DeltaExtended{F}) where {N, F <: NTuple{N, Function}} = DeltaUEKnownInverseApproximationDeltaFnRuleLayout()
