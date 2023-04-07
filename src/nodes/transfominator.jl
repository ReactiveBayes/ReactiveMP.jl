export transfominator, Transfominator, TMeta

import LazyArrays, BlockArrays
import StatsFuns: log2π

struct Transfominator end

const transfominator = Transfominator

@node Transfominator Stochastic [y, x, h, Λ]

struct TMeta
    ds::Tuple # dimensionality of Transfominator (dy, dx)
    Fs::Vector{<:AbstractMatrix} # masks
    es::Vector{<:AbstractVector} # unit vectors

    function TMeta(ds::Tuple{T, T}) where {T <: Integer}
        dy, dx = ds
        Fs = [tmask(dx, dy, i) for i in 1:dy]
        es = [StandardBasisVector(dy, i, one(T)) for i in 1:dy]
        return new(ds, Fs, es)
    end

    function TMeta(dy::T, dx::T) where {T <: Integer}
        Fs = [tmask(dx, dy, i) for i in 1:dy]
        es = [StandardBasisVector(dy, i, one(T)) for i in 1:dy]
        return new((dy, dx), Fs, es)
    end
end

@average_energy Transfominator (q_y_x::MultivariateNormalDistributionsFamily, q_h::MultivariateNormalDistributionsFamily, q_Λ::Wishart, meta::TMeta) = begin
    mh, Vh   = mean_cov(q_h)
    myx, Vyx = mean_cov(q_y_x)
    mΛ       = mean(q_Λ)

    dy, dx = getdimensionality(meta)
    Fs, es = getmasks(meta), getunits(meta)
    n      = div(ndims(q_y_x), 2)
    mH     = tcompanion_matrix(mh, meta)
    mx, Vx = myx[(dy + 1):end], Vyx[(dy + 1):end, (dy + 1):end]
    my, Vy = myx[1:dy], Vyx[1:dy, 1:dy]
    Vyx    = Vyx[1:dy, (dy + 1):end]
    g₁     = my' * mΛ * my + tr(Vy * mΛ)
    g₂     = mx' * mH' * mΛ * my + tr(Vyx * mH' * mΛ)
    g₃     = g₂
    G      = sum(sum(es[i]' * mΛ * es[j] * Fs[i] * (mh * mh' + Vh) * Fs[j]' for i in 1:length(Fs)) for j in 1:length(Fs))
    g₄     = mx' * G * mx + tr(Vx * G)
    AE     = n / 2 * log2π - 0.5 * mean(logdet, q_Λ) + 0.5 * (g₁ - g₂ - g₃ + g₄)

    return AE
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
    dy, dx = getdimensionality(meta)
    L = sum(es[i] * w' * Fs[i]' for i in 1:dy)
    return L
end
