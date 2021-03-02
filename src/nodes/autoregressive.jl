export AR, ARsafe, ARunsafe, ARMeta

struct AR end

struct ARsafe end
struct ARunsafe end

struct ARMeta{F <: VariateForm, S}
    order :: Int
end

function ARMeta(order, ::Type{F}, ::Type{S}) where {F, S}
    return ARMeta{F, S}(order)
end

@node AR Stochastic [ y, x, θ, γ ]

@average_energy AR (q_y_x::MultivariateNormalDistributionsFamily, q_θ::NormalDistributionsFamily, q_γ::GammaShapeRate, meta::ARMeta{F}) where F = begin
    mθ, Vθ   = mean(q_θ), cov(q_θ)
    myx, Vyx = mean(q_y_x), cov(q_y_x)
    mγ       = mean(q_γ)

    mx, Vx   = myx[end], Vyx[end]
    my1, Vy1 = myx[1], Vyx[1]
    Vy1x     = F == Multivariate ? Vyx[1, order+1:end] : Vyx[3]

    -0.5*(logmean(q_γ)) +
    0.5*log(2*pi) + 0.5*mγ*(Vy1+my1^2 - 2*mθ'*(Vy1x + mx*my1) +
    tr(Vθ*Vx) + mx'*Vθ*mx + mθ'*(Vx + mx*mx')*mθ)

    # correction
    if F == Multivariate
        AE += entropy(marg_y_x)
        idc = [1, order+1:2*order...]
        myx_n = myx[idc]
        Vyx_n = Vyx[idc, idc]
        marg_y_x = MvNormalMeanCovariance(myx_n, Vyx_n)
        AE -= entropy(marg_y_x)
    end
end
