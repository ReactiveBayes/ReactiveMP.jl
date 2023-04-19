export GaussianProcess, ProcessMeta 

# Create GP structure 
struct GaussianProcess{F, K, M, I1, I2, I4, S}
    meanfunction        ::F
    kernelfunction      ::K
    finitemarginal      ::M  
    testinput           ::I1
    traininput          ::I2
    inducing_input      ::I4
    covariance_strategy ::S
end

#ProcessMeta to store the index of observation 
struct ProcessMeta 
    index :: Int 
end 


@node GaussianProcess Stochastic [out, meanfunc, kernelfunc, params]

### compute free energy here 
@average_energy GaussianProcess (q_out::GaussianProcess, q_meanfunc::Any, q_kernelfunc::Any, q_params::Any) = begin
    θ = mean(q_params)
    y, Σ = mean_cov(q_out.finitemarginal)
    xtest = q_out.testinput
    kernel = q_kernelfunc
    meanfunc = q_meanfunc 
    xu = q_out.inducing_input
    strategy = q_out.covariance_strategy
    return gp_avg_energy(strategy,meanfunc,kernel,Σ,y,xtest,θ,xu)
end

@average_energy GaussianProcess (q_out::GaussianProcess, q_meanfunc::Any, q_kernelfunc::Any, q_params::Any, meta::ProdCVI) = begin
    θ = mean(q_params)
    y, Σ = mean_cov(q_out.finitemarginal)
    xtest = q_out.testinput
    kernel = q_kernelfunc
    meanfunc = q_meanfunc 
    xu = q_out.inducing_input
    strategy = q_out.covariance_strategy
    return gp_avg_energy(strategy,meanfunc,kernel,Σ,y,xtest,θ,xu)
end

# function Distributions.entropy(pm::PointMass{F}) where {F <: Function}
#     return ReactiveMP.CountingReal(Float64,-1)
# end

# function Distributions.entropy(pm::PointMass{F}) where {F <: Kernel}
#     return ReactiveMP.CountingReal(Float64,-1)
# end

function Distributions.entropy(pm::Any)
    return ReactiveMP.CountingReal(Float64,-1)
end

function ReactiveMP.entropy(p::GaussianProcess)
    return ReactiveMP.entropy(p.finitemarginal)
end
function mean(p::GaussianProcess)
    return mean(p.finitemarginal)
end
function cov(p::GaussianProcess)
    return cov(p.finitemarginal)
end
function gp_avg_energy end 

function gp_avg_energy(::CovarianceMatrixStrategy{<:FullCovarianceStrategy},meanfunc, kernel, Σ_finitemarginal, y_finitemarginal, x, θ, x_inducing)
    N = length(y_finitemarginal)
    K = kernelmatrix(kernel(exp.(θ)),x,x)
    m = meanfunc.(x)
    return  1/2 * (tr(cholinv(K) * Σ_finitemarginal) + y_finitemarginal' * cholinv(K) * y_finitemarginal) - m'*cholinv(K)*y_finitemarginal - 1/2 * m' * cholinv(K) * m + chollogdet(K) + N*log(2π)/2
end
#### The below functions give incorrect FE for SoR, DTC and FITC
function gp_avg_energy(::CovarianceMatrixStrategy{<:DeterministicInducingConditional},meanfunc,kernel, Σ_finitemarginal, y_finitemarginal, x, θ, x_inducing)
    N = length(y_finitemarginal)
    Kuu = kernelmatrix(kernel(exp.(θ)),x_inducing, x_inducing)
    Kfu = kernelmatrix(kernel(exp.(θ)),x,x_inducing)
    Qff = Kfu * cholinv(Kuu) * Kfu'
    m = meanfunc.(x)

    return  1/2 * (tr(cholinv(Qff) * Σ_finitemarginal) + y_finitemarginal' * cholinv(Qff) * y_finitemarginal) - m'*cholinv(Qff)*y_finitemarginal - 1/2 * m' * cholinv(Qff) * m + chollogdet(Qff) + N*log(2π)/2 
end

function gp_avg_energy(::CovarianceMatrixStrategy{<:DeterministicTrainingConditional},meanfunc,kernel, Σ_finitemarginal, y_finitemarginal, x, θ, x_inducing)
    N = length(y_finitemarginal)
    Kuu = kernelmatrix(kernel(exp.(θ)),x_inducing, x_inducing)
    Kfu = kernelmatrix(kernel(exp.(θ)),x,x_inducing)
    Qff = Kfu * cholinv(Kuu) * Kfu'
    m = meanfunc.(x)
    return  1/2 * (tr(cholinv(Qff) * Σ_finitemarginal) + y_finitemarginal' * cholinv(Qff) * y_finitemarginal) - m'*cholinv(Qff)*y_finitemarginal - 1/2 * m' * cholinv(Qff) * m + chollogdet(Qff) + N*log(2π)/2
end

function gp_avg_energy(::CovarianceMatrixStrategy{<:FullyIndependentTrainingConditional},meanfunc,kernel, Σ_finitemarginal, y_finitemarginal, x, θ, x_inducing)
    N = length(y_finitemarginal)
    Kuu = kernelmatrix(kernel(exp.(θ)),x_inducing, x_inducing)
    Kff = kernelmatrix(kernel(exp.(θ)),x,x)
    Kfu = kernelmatrix(kernel(exp.(θ)),x,x_inducing)
    Qff = Kfu * cholinv(Kuu) * Kfu'
    Λ = Diagonal(Kff - Qff)
    return  1/2 * (tr(cholinv(Qff + Λ) * Σ_finitemarginal) + y_finitemarginal' * cholinv(Qff + Λ) * y_finitemarginal) + chollogdet(Qff + Λ) + N*log(2π)/2
end