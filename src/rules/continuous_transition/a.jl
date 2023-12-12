@rule ContinuousTransition(:a, Marginalisation) (q_y_x::MultivariateNormalDistributionsFamily, q_a::MultivariateNormalDistributionsFamily, q_W::Any, meta::CTMeta) = begin
    ma = mean(q_a)
    mW = mean(q_W)
    myx, Vyx = mean_cov(q_y_x)

    Fs = getjacobians(meta, ma)
    dy = length(Fs)

    mx, Vx = @views myx[(dy + 1):end], Vyx[(dy + 1):end, (dy + 1):end]
    my, Vy = @views myx[1:dy], Vyx[1:dy, 1:dy]
    Vyx    = @view Vyx[1:dy, (dy + 1):end]

    xi, W  =  zeros(eltype(ma), length(ma)), zeros(eltype(ma), length(ma), length(ma))

    Vxymxy = rank1update(Vyx', mx, my)
    Vxmx = rank1update(Vx, mx)
    for i in 1:dy
        xi += Fs[i]' * Vxymxy * mW[:,i]
        for j in 1:dy
            W += mW[j,i] * Fs[i]'*Vxmx*Fs[j]
        end
    end
    
    return MvNormalWeightedMeanPrecision(xi, W)
end
