export transfominator, Transfominator, TMeta

import LazyArrays, BlockArrays
import StatsFuns: log2π

struct Transfominator end

const transfominator = Transfominator

struct TMeta
    ds :: Tuple # dimensionality of Transfominator
    Fs :: Vector{<:AbstractMatrix} # masks
    es :: Vector{<:AbstractVector} # unit vectors

    function TMeta(ds::Tuple{T, T}) where {T}
        dim1, dim2 = ds
        Fs = [tmask(dim1, dim2, i) for i in 1:dim2]
        es = [StandardBasisVector(dim2, i, one(T)) for i in 1:dim2]
        return new(ds, Fs, es)
    end
end

getdimensionality(meta::TMeta) = meta.ds
getmasks(meta::TMeta)          = meta.Fs
getunits(meta::TMeta)          = meta.es

@node Transfominator Stochastic [y, x, w, Λ]

default_meta(::Type{TMeta}) = error("Transfominator node requires meta flag explicitly specified")

function tmask(dim1, dim2, index)
    F = zeros(dim1, dim1 * dim2)
    start_col = (index - 1) * dim1 + 1
    end_col = start_col + dim1 - 1
    @inbounds F[1:dim1, start_col:end_col] = I(dim1)
    return F
end

function tcompanion_matrix(w, meta::TMeta)
    Fs, es = getmasks(meta), getunits(meta)
    L = sum(es[i] * w' * Fs[i]' for i in 1:dim2)
    return L
end
