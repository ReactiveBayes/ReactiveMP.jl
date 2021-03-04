export AR, ARsafe, ARunsafe, ARMeta

struct AR end

const Autoregressive = AR

struct ARsafe end
struct ARunsafe end

struct ARMeta{F <: VariateForm, S}
    order :: Int
end

function ARMeta(::Type{Univariate}, ::Type{S}) where S
    return ARMeta{Univariate, S}(1)
end

function ARMeta(::Type{Multivariate}, order, ::Type{S}) where S
    return ARMeta{Multivariate, S}(order)
end

@node AR Stochastic [ y, x, θ, γ ]

@average_energy AR (q_y_x::MultivariateNormalDistributionsFamily, q_θ::NormalDistributionsFamily, q_γ::GammaShapeRate, meta::ARMeta{F}) where F = begin
    mθ, Vθ   = mean(q_θ), cov(q_θ)
    myx, Vyx = mean(q_y_x), cov(q_y_x)
    mγ       = mean(q_γ)

    order = meta.order

    mx, Vx   = arslice(F, myx, order+1:2order), arslice(F, Vyx, order+1:2order, order+1:2order)
    my1, Vy1 = myx[1], Vyx[1]
    Vy1x     = arslice(F, Vyx, 1, order+1:2order)
    
    AE = -0.5*(logmean(q_γ)) +
    0.5*log(2*pi) + 0.5*mγ*(Vy1+my1^2 - 2*mθ'*(Vy1x + mx*my1) +
    tr(Vθ*Vx) + mx'*Vθ*mx + mθ'*(Vx + mx*mx')*mθ)

    # correction
    if F == Multivariate
        AE += entropy(q_y_x)
        idc = [1, order+1:2order...]
        myx_n = myx[idc]
        Vyx_n = Vyx[idc, idc]
        q_y_x = MvNormalMeanCovariance(myx_n, Vyx_n)
        AE -= entropy(q_y_x)
    end
    return AE
end

# Helpers for AR rules
function arslice(::Type{Multivariate}, array, ranges...)
    return array[ranges...]
end

function arslice(::Type{Univariate}, array, ranges...)
    return first(array[ranges...])
end

function uvector(::Type{Multivariate}, order)
    c = zeros(order); c[1] = 1.0
    return c
end

uvector(::Type{Univariate}, order) = 1.0
