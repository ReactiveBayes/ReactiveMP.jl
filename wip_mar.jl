using RxInfer
using LinearAlgebra
using Plots
using Random

Random.seed!(42)

function generate_mar(order, ds, n_samples)
    As = [0.1*randn(ds,ds) for _ in 1:order]

    x = [randn(ds) for _ in 1:order]
    y = deepcopy(x)

    for _ in 1:n_samples
        m = mapreduce(x -> x[1] * x[2], +, zip(As, x[end:-1:end-order+1]))
        dist = MvNormal(m, diageye(ds))
        push!(x, rand(dist))
        push!(y, rand(MvNormal(x[end], diageye(ds))))
    end
    collect(Iterators.flatten(hcat(As))), x, y
end

@model function multivariateAR(n_samples, order, dimension)

    o = datavar(Vector{Float64}, n_samples)
    y = randomvar(n_samples)

    a ~ MvNormalMeanCovariance(randn(dimension^2*order), diageye(dimension^2*order))

    # NOTE: Wishart is naughty
    Λ ~ Wishart(dimension, diageye(dimension))

    B = zeros(dimension, dimension*order); B[1:dimension, 1:dimension] = diageye(dimension)

    x ~ MvNormalMeanCovariance(zeros(dimension*order), diageye(dimension*order))
    
    x_prev = x
    for i in 1:n_samples
        y[i] ~ MAR(x_prev, a, Λ) where {meta = MARMeta(order, dimension)}
        
        o[i] ~ MvNormalMeanCovariance(B*y[i], diageye(dimension))
        x_prev = y[i]
    end
end

constraints = @constraints begin 
    q(y, x, a, Λ) = q(y, x)q(a)q(Λ)
end

# constraints = @constraints begin 
#     q(y, x, a, Λ, τ) = q(y, x)q(a)q(Λ)q(τ)
# end

# mf_constraints = @constraints begin 
#     q(y, x, a, Λ) = q(y)q(x)q(a)q(Λ)
# end

n = 100

d = 4
p = 5
coefs, lat, obs = generate_mar(p, d, n)

transform_obs = [vcat(obs[i], obs[i-1]) for i in 2:n+1] 
corrected_obs = [x[1:d] for x in transform_obs]

mdata          = (o = corrected_obs, )
minitmarginals = (Λ = Wishart(d, diageye(d)), a = MvNormalMeanPrecision(zeros(d^2*p), diageye(d^2*p)))
minitmarginals = (Λ = Wishart(d, diageye(d)), a = MvNormalMeanPrecision(zeros(d^2*p), diageye(d^2*p)), x = vague(MvNormalMeanCovariance, d*p), y = vague(MvNormalMeanCovariance, d*p))

# minitmarginals = (Λ = Wishart(d+1, diageye(d)), a = MvNormalMeanPrecision(zeros(d^2*p), diageye(d^2*p)), τ = GammaShapeRate(1.0, 1.0))

# First execution is slow due to Julia's initial compilation 
mresult = inference(
    model = multivariateAR(n, p, d), 
    data  = mdata,
    # constraints   = mf_constraints,
    constraints   = constraints,
    initmarginals = minitmarginals,
    free_energy   = true,
    iterations    = 25, 
    showprogress  = true,
)


scatter(coefs)
plot!(mean(mresult.posteriors[:a][end]), ribbon=sqrt.(var(mresult.posteriors[:a][end])))
vline!([i*d*d for i in 1:p])

plot(mresult.free_energy[1:end])

lat_mean = mean.(mresult.posteriors[:y][end])
lat_var = var.(mresult.posteriors[:y][end])

lat_mean₁ = first.(lat_mean)
lat_mean₂ = getindex.(lat_mean, 2)
lat_mean₃ = getindex.(lat_mean, 3)
lat_mean₄ = getindex.(lat_mean, 4)

lat_var₁ = first.(lat_var)
lat_var₂ = getindex.(lat_var, 2)
lat_var₃ = getindex.(lat_var, 3)
lat_var₄ = getindex.(lat_var, 4)

Λ_inf = mresult.posteriors[:Λ][end]
@show mean(Λ_inf)

scatter(first.(mdata[:o]))
plot!(first.(lat[2:end]))
plot!(lat_mean₁, ribbon=sqrt.(lat_var₁))

scatter(getindex.(mdata[:o], 2))
plot!(getindex.(lat[2:end], 2))
plot!(lat_mean₂, ribbon=sqrt.(lat_var₂))

scatter(getindex.(mdata[:o], 3))
plot!(getindex.(lat[2:end], 3))
plot!(lat_mean₃, ribbon=sqrt.(lat_var₃))

scatter(getindex.(mdata[:o], 4))
plot!(getindex.(lat[2:end], 4))
plot!(lat_mean₄, ribbon=sqrt.(lat_var₄))


