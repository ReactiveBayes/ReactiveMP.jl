export GeneralizedTProcess

# Create GP structure 
struct GeneralizedTProcess{F, K, V, M,I1, I2, I4, S}
    meanfunction        ::F
    kernelfunction      ::K
    degree              ::V
    finitemarginal      ::M  
    testinput           ::I1
    traininput          ::I2
    inducing_input      ::I4
    covariance_strategy ::S
end


@node GeneralizedTProcess Stochastic [out, meanfunc, kernelfunc, degree, params]

### compute free energy here 
@average_energy GeneralizedTProcess (q_out::GeneralizedTProcess, q_meanfunc::Any, q_kernelfunc::Any, q_degree::Any,q_params::Any) = begin
    θ = mean(q_params)
    y, Σ = mean_cov(q_out.finitemarginal)
    xtest = q_out.testinput
    kernel = q_kernelfunc
    meanfunc = q_meanfunc 
    xu = q_out.inducing_input
    strategy = q_out.covariance_strategy
    return tp_avg_energy(strategy,meanfunc,kernel,Σ,y,xtest,θ,xu)
end

@average_energy GeneralizedTProcess (q_out::GeneralizedTProcess, q_meanfunc::Any, q_kernelfunc::Any,q_degree::Any, q_params::Any, meta::ProdCVI) = begin
    θ = mean(q_params)
    y, Σ = mean_cov(q_out.finitemarginal)
    xtest = q_out.testinput
    kernel = q_kernelfunc
    meanfunc = q_meanfunc 
    xu = q_out.inducing_input
    strategy = q_out.covariance_strategy
    return tp_avg_energy(strategy,meanfunc,kernel,Σ,y,xtest,θ,xu)
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

function ReactiveMP.entropy(p::GeneralizedTProcess)
    return ReactiveMP.entropy(p.finitemarginal)
end
function mean(p::GeneralizedTProcess)
    return mean(p.finitemarginal)
end
function cov(p::GeneralizedTProcess)
    return cov(p.finitemarginal)
end
function tp_avg_energy end 

function tp_avg_energy(::CovarianceMatrixStrategy{<:FullCovarianceStrategy},meanfunc, kernel, Σ_finitemarginal, y_finitemarginal, x, θ, x_inducing)
    N = length(y_finitemarginal)
    K = kernelmatrix(kernel(exp.(θ)),x,x)
    m = meanfunc.(x)
    return  1/2 * (tr(cholinv(K) * Σ_finitemarginal) + y_finitemarginal' * cholinv(K) * y_finitemarginal) - m'*cholinv(K)*y_finitemarginal - 1/2 * m' * cholinv(K) * m + chollogdet(K) + N*log(2π)/2
end
