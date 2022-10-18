export DeltaLinearization

Base.@kwdef struct DeltaLinearization{T}
    inverse::T = nothing
end

deltafn_rule_layout(::DeltaFnNode, ::DeltaLinearization{Nothing}) =
    DeltaUknownInverseApproximationDeltaFnRuleLayout()

deltafn_rule_layout(::DeltaFnNode, ::DeltaLinearization{F}) where {F <: Function}               = DeltaKnownInverseApproximationDeltaFnRuleLayout()
deltafn_rule_layout(::DeltaFnNode, ::DeltaLinearization{F}) where {N, F <: NTuple{N, Function}} = DeltaKnownInverseApproximationDeltaFnRuleLayout()
