# NOTE: Both rules require q_a as input. This is a particular requirement for the ContinuousTransition node as it might need the expansion point for the transformation. This is not a general requirement for the VMP rules.

# VMP: Stuctured
@rule ContinuousTransition(:a, Marginalisation) (q_y_x::MultivariateNormalDistributionsFamily, q_a::MultivariateNormalDistributionsFamily, q_W::Any, meta::CTMeta) = begin
    ma = mean(q_a)
    mW = mean(q_W)
    myx, Vyx = mean_cov(q_y_x)

    Fs = getjacobians(meta, ma)
    dy = length(Fs)

    mx, Vx = @views myx[(dy + 1):end], Vyx[(dy + 1):end, (dy + 1):end]
    my, Vy = @views myx[1:dy], Vyx[1:dy, 1:dy]
    Vyx    = @view Vyx[1:dy, (dy + 1):end]

    xi, W = zeros(eltype(ma), length(ma)), zeros(eltype(ma), length(ma), length(ma))

    Vxymxy = rank1update(Vyx', mx, my)
    Vxmx = rank1update(Vx, mx)
    for i in 1:dy
        xi += Fs[i]' * Vxymxy * mW[:, i]
        for j in 1:dy
            W += mW[j, i] * Fs[i]' * Vxmx * Fs[j]
        end
    end

    return MvNormalWeightedMeanPrecision(xi, W)
end

# VMP: Mean-field
@rule ContinuousTransition(:a, Marginalisation) (q_y::Any, q_x::Any, q_a::Any, q_W::Any, meta::CTMeta) = begin
    mx, Vx = mean_cov(q_x)
    mW = mean(q_W)
    my = mean(q_y)
    ma = mean(q_a)

    Fs = getjacobians(meta, ma)
    dy = length(Fs)

    xi, W = zeros(eltype(ma), length(ma)), zeros(eltype(ma), length(ma), length(ma))

    mxmy = mx * my'
    Vxmx = rank1update(Vx, mx)

    for i in 1:dy
        xi += Fs[i]' * mxmy * mW[:, i]
        for j in 1:dy
            W += mW[j, i] * Fs[i]' * Vxmx * Fs[j]
        end
    end

    return MvNormalWeightedMeanPrecision(xi, W)
end
