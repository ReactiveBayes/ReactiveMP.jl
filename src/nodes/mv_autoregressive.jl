export MAR, MvAutoregressive, MARMeta, mar_transition, mar_shift

import LazyArrays, BlockArrays
import StatsFuns: log2π

struct MAR end

const MvAutoregressive = MAR

struct MARMeta
    order :: Int # order (lag) of MAR
    ds    :: Int # dimensionality of MAR process, i.e., the number of correlated AR processes

    function MARMeta(order, ds=2)
        if ds < 2
            @error "ds parameter should be > 1. Use AR node if ds = 1"
        end
        return new(order, ds)
    end
end


getorder(meta::MARMeta)              = meta.order
getdimensionality(meta::MARMeta)     = meta.ds

@node MAR Stochastic [y, x, a, Λ]

default_meta(::Type{MAR}) = error("MvAutoregressive node requires meta flag explicitly specified")

@average_energy MAR (
    q_y_x::MultivariateNormalDistributionsFamily,
    q_a::MultivariateNormalDistributionsFamily,
    q_Λ::Wishart,
    meta::MARMeta
) = begin
    ma, Va   = mean_cov(q_a)
    myx, Vyx = mean_cov(q_y_x)
    mΛ       = mean(q_Λ)

    order, ds = getorder(meta), getdimensionality(meta)
    F = Multivariate
    dim = order*ds
    n = div(ndims(q_y_x), 2)


    ma, Va = mean_cov(q_a)
    mA = mar_companion_matrix(order, ds, ma)[1:ds, 1:dim]

    mx, Vx   = ar_slice(F, myx, (dim+1):2dim), ar_slice(F, Vyx, (dim+1):2dim, (dim+1):2dim)
    my1, Vy1 = myx[1:ds], Vyx[1:ds, 1:ds]
    Vy1x     = ar_slice(F, Vyx, 1:ds, dim+1:2dim)

    # @show Vyx
    # @show Vy1x

    # this should be inside MARMeta
    es = [uvector(ds, i) for i in 1:ds]
    Fs = [mask_mar(order, ds, i) for i in 1:ds]

    g₁ = my1'*mΛ*my1 + tr(Vy1*mΛ)
    g₂ = mx'*mA'*mΛ*my1 + tr(Vy1x*mA'*mΛ)
    g₃ = g₂
    G = sum(sum(es[i]'*mΛ*es[j]*Fs[i]*(ma*ma' + Va)*Fs[j]' for i in 1:ds) for j in 1:ds)
    g₄ = mx'*G*mx + tr(Vx*G)
    AE =  n/2*log2π - 0.5*mean(logdet, q_Λ) + 0.5*(g₁ - g₂ - g₃ + g₄)

    if order > 1
        AE += entropy(q_y_x)
        idc = LazyArrays.Vcat(1:ds, dim+1:2dim)
        myx_n = view(myx, idc)
        Vyx_n = view(Vyx, idc, idc)
        q_y_x = MvNormalMeanCovariance(myx_n, Vyx_n)
        AE -= entropy(q_y_x)
    end

    return AE
end

@average_energy MAR (
    q_y::MultivariateNormalDistributionsFamily,
    q_x::MultivariateNormalDistributionsFamily,
    q_a::MultivariateNormalDistributionsFamily,
    q_Λ::Wishart,
    meta::MARMeta
) = begin
    ma, Va   = mean_cov(q_a)
    my, Vy = mean_cov(q_y)
    mx, Vx = mean_cov(q_y)
    mΛ       = mean(q_Λ)

    order, ds = getorder(meta), getdimensionality(meta)
    F = Multivariate
    dim = order*ds
    n = dim

    ma, Va = mean_cov(q_a)
    mA = mar_companion_matrix(order, ds, ma)[1:ds, 1:dim]

    my1, Vy1 = my[1:ds], Vy[1:ds, 1:ds]

    # this should be inside MARMeta
    es = [uvector(ds, i) for i in 1:ds]
    Fs = [mask_mar(order, ds, i) for i in 1:ds]

    g₁ = my1'*mΛ*my1 + tr(Vy1*mΛ)
    g₂ = -mx'*mA'*mΛ*my1
    g₃ = -g₂
    G = sum(sum(es[i]'*mΛ*es[j]*Fs[i]*(ma*ma' + Va)*Fs[j]' for i in 1:ds) for j in 1:ds)
    g₄ = mx'*G*mx + tr(Vx*G)
    AE =  n/2*log2π - 0.5*mean(logdet, q_Λ) + 0.5*(g₁ + g₂ + g₃ + g₄)

    if order > 1
        AE += entropy(q_y)
        q_y = MvNormalMeanCovariance(my1, Vy1)
        AE -= entropy(q_y)
    end

    return AE
end

# Helpers for AR rules
function mask_mar(order, dimension, index)
    F = zeros(dimension*order, dimension*dimension*order)
    rows = repeat([dimension], order)
    cols = repeat([dimension], dimension*order)
    FB = BlockArrays.BlockArray(F, rows, cols)
    for k in 1:order
        for j in 1:dimension*order
            if j == index + (k-1)*dimension
                view(FB, BlockArrays.Block(k, j)) .= diageye(dimension)
            end
        end
    end
    return Matrix(FB)
end

function mar_transition(order, Λ)
    dim = size(Λ, 1)
    W = 1.0*diageye(dim*order)
    W[1:dim, 1:dim] = Λ
    return W
end


function mar_shift(order, ds)
    dim = order*ds
    S = diageye(dim)
    for i in dim:-1:ds+1
        S[i,:] = S[i-ds, :]
    end
    S[1:ds, :] = zeros(ds, dim)
    return S
end

function uvector(dim, pos=1)
    u = zeros(dim)
    u[pos] = 1
    return dim == 1 ? u[pos] : u
end

function mar_companion_matrix(order, ds, a)
    dim = order*ds
    S = mar_shift(order, ds)
    es = [uvector(dim, i) for i in 1:ds]
    Fs = [mask_mar(order, ds, i) for i in 1:ds]
    L =  S .+ sum(es[i]*a'*Fs[i]' for i in 1:ds)
    return L
end
