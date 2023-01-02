export MAR, MvAutoregressive, MARMeta, mar_transition, mar_shift

import LazyArrays, BlockArrays
import StatsFuns: log2π

struct MAR end

const MvAutoregressive = MAR

struct MARMeta
    order :: Int # order (lag) of MAR
    ds    :: Int # dimensionality of MAR process, i.e., the number of correlated AR processes
end

function MARMeta(order, ds=2)
    if ds < 2
        @error "ds parameter should be > 1. Use AR node if ds = 1"
    end
    return MARMeta(order, ds)
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
    mA = mar_companion_matrix(order, ds, ma)[1:order, 1:dim]

    mx, Vx   = ar_slice(F, myx, (dim+1):2dim), ar_slice(F, Vyx, (dim+1):2dim, (dim+1):2dim)
    my1, Vy1 = myx[1:ds], Vyx[1:ds, 1:ds]
    Vy1x     = ar_slice(F, Vyx, 1:ds, dim+1:2dim)

    # this should be inside MARMeta
    es = [uvector(order, i) for i in 1:order]
    Fs = [mask_mar(order, ds, i) for i in 1:order]

    # # Euivalento to AE = (-mean(log, q_γ) + log2π + mγ*(Vy1+my1^2 - 2*mθ'*(Vy1x + mx*my1) + tr(Vθ*Vx) + mx'*Vθ*mx + mθ'*(Vx + mx*mx')*mθ)) / 2
    g₁ = my1'*mΛ*my1 + tr(Vy1*mΛ)
    g₂ = -mx'*mA'*mΛ*my1 + tr(Vy1x*mA'*mΛ)
    g₃ = -g₂
    G = sum(sum(es[i]'*mΛ*es[j]*Fs[i]*(ma*ma' + Va)*Fs[j]' for i in 1:order) for j in 1:order)
    g₄ = mx'*G*mx + tr(Vx*G)
    AE =  n/2*log2π - 0.5*mean(logdet, q_Λ) + 0.5*(g₁ + g₂ + g₃ + g₄)

    if order > 1
        mean(q_y_x)
        AE += entropy(q_y_x)
        idc = LazyArrays.Vcat(1:order, (dim+1):2dim)
        myx_n = view(myx, idc)
        Vyx_n = view(Vyx, idc, idc)
        q_y_x = MvNormalMeanCovariance(myx_n, Vyx_n)
        AE -= entropy(q_y_x)
    end

    return AE
end

# Helpers for AR rules


# p, d, i
# function mask_mar(order, ds, index)
#     Frows = order * ds
#     Fcols = ds * Frows
#     F = zeros(Frows, Fcols)
#     FB = BlockArray(F, )
#     for k in 1:ds*order
#         for j in 1:ds*order^2
#             if j == ds*(index+(k-1)*ds)
#                 F[k, j] = 1.0
#             else
#                 F[k, j] = .0
#             end
#         end
#         end
#     # F[1:order, ds*index-1:ds*index] = diageye(ds)
#     # F[ds+1:end, ds*index+ds+1:ds*index+ds+ds] = diageye(ds)
#     @show F
#     return F
# end

function mask_mar(p, d, index)
    F = zeros(d*p, d*d*p)
    rows = repeat([d], p)
    cols = repeat([d], d*p)
    FB = BlockArrays.BlockArray(F, rows, cols)
    for k in 1:p
        for j in 1:d*p
            if j == index + (k-1)*d
                view(FB, BlockArrays.Block(k, j)) .= diageye(d)
            end
        end
    end
    return Matrix(FB)
end

function mar_transition(order, Λ)
    dim = size(Λ, 1)
    W = 1e12*diageye(dim*order)
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
    es = [uvector(dim, i) for i in 1:order]
    Fs = [mask_mar(order, ds, i) for i in 1:order]
    return S .+ sum(es[i]*a'*Fs[i]' for i in 1:order)
end