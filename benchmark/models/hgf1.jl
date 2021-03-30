# Hierarchical Gaussian Filter

module HGF1Benchmark

using Rocket
using ReactiveMP
using GraphPPL
using Distributions

@model function hgf()

    real_k     = constvar(1.0)
    real_w     = constvar(-5.0)
    y_variance = constvar(0.01)
    z_variance = constvar(1.0)
    
    xt_min_mean = datavar(Float64)
    xt_min_var  = datavar(Float64)
    
    zt_min_mean = datavar(Float64)
    zt_min_var  = datavar(Float64)
    
    xt_min ~ NormalMeanVariance(xt_min_mean, xt_min_var)
    zt_min ~ NormalMeanVariance(zt_min_mean, zt_min_var)
    
    zt ~ NormalMeanVariance(zt_min, z_variance) where { q = q(zt_min)q(z_variance)q(zt) }
    
    gcv_node, xt ~ GCV(xt_min, zt, real_k, real_w) where { q = q(xt, xt_min)q(zt)q(κ)q(ω) }
    
    y = datavar(Float64)
    
    y ~ NormalMeanVariance(xt, y_variance)
    
    return zt, xt, y, gcv_node, xt_min_mean, xt_min_var, zt_min_mean, zt_min_var
end

function generate_input(rng, n)
    real_k = 1.0
    real_w = -5.0

    z_prev = 0.0
    x_prev = 0.0

    z = Vector{Float64}(undef, n)
    v = Vector{Float64}(undef, n)
    x = Vector{Float64}(undef, n)
    y = Vector{Float64}(undef, n)

    y_variance = 0.01
    z_variance = 1.0

    for i in 1:n
        z[i] = rand(rng, Normal(z_prev, sqrt(z_variance)))
        v[i] = exp(real_k * z[i] + real_w)
        x[i] = rand(rng, Normal(x_prev, sqrt(v[i])))
        y[i] = rand(rng, Normal(x[i], sqrt(y_variance)))
        
        z_prev = z[i]
        x_prev = x[i]
    end

    return y
end

function benchmark(input)
    iters      = 10
    real_k     = 1.0
    real_w     = -5.0
    y_variance = 0.01
    z_variance = 1.0

    n = length(input)
    
    ms_scheduler = PendingScheduler()
    fe_scheduler = PendingScheduler()
    
    mz = keep(Marginal)
    mx = keep(Marginal)
    fe = keep(Float64)

    model, (zt, xt, y, gcv_node, xt_min_mean, xt_min_var, zt_min_mean, zt_min_var) = hgf()

    s_mz = subscribe!(getmarginal(zt) |> schedule_on(ms_scheduler), mz)
    s_mx = subscribe!(getmarginal(xt) |> schedule_on(ms_scheduler), mx)
    s_fe = subscribe!(score(Float64, BetheFreeEnergy(), model, fe_scheduler), fe)
    
    # Initial prior messages
    current_zt = NormalMeanVariance(0.0, 10.0)
    current_xt = NormalMeanVariance(0.0, 10.0)

    # Prior marginals
    setmarginal!(gcv_node, :y_x, MvNormalMeanCovariance([ 0.0, 0.0 ], [ 5.0, 5.0 ]))
    setmarginal!(gcv_node, :z, NormalMeanVariance(0.0, 5.0))
    
    for i in 1:n
        
        for _ in 1:iters
            update!(y, input[i])
            update!(zt_min_mean, mean(current_zt))
            update!(zt_min_var, var(current_zt))
            update!(xt_min_mean, mean(current_xt))
            update!(xt_min_var, var(current_xt))
            
            release!(fe_scheduler)
        end
        
        release!(ms_scheduler)
        
        current_zt = mz[end]
        current_xt = mx[end]
    end
    
    unsubscribe!(s_mz)
    unsubscribe!(s_mx)
    unsubscribe!(s_fe)
    
    return map(getvalues, (mz, mx, fe))
end

end