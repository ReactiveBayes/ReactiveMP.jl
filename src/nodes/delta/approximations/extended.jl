export DeltaExtended, ET

struct DeltaExtended{T}
    inverse :: T
end

DeltaExtended() = DeltaExtended(nothing)

const ET = DeltaExtended