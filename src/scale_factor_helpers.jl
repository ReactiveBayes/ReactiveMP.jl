export ScaleFactorMeta, ScaledMessage

abstract type AbstractMeta end

struct ScaleFactorMeta <: AbstractMeta end

struct ScaledMessage{T}
    message :: T
    scale   :: Float64
end