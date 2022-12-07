export GaussianProcess, ProcessMeta 

# Create GP structure 
struct GaussianProcess
    meanfunction        ::Function 
    kernelfunction      ::Kernel
    finitemarginal      
    testinput           ::Array{Float64, 1}    
    traininput          ::Array{Float64, 1}
    observation         ::Array{Float64, 1}
    observation_noise   ::Array{Float64, 2}     
    inducing_input      ::Array{Float64, 1}
    covariance_strategy ::AbstractCovarianceStrategyType
end

struct ProcessMeta 
    index :: Int 
end 

@node GaussianProcess Stochastic [out, meanfunc, kernelfunc, params]

### compute free energy here 
@average_energy GaussianProcess (q_out::GaussianProcess, q_meanfunc::PointMass, q_kernelfunc::PointMass, q_params::Any) = begin
    θ = mean(q_params)
    y, Σ = mean_cov(q_out.finitemarginal)
    xtest = q_out.testinput
    kernel = q_kernelfunc.point
    meanfunc = q_meanfunc.point 
    K = kernelmatrix(kernel(exp.(θ)),xtest,xtest)
    N = length(xtest)
    return ((y - meanfunc.(xtest))' * cholinv(K + Σ) * (y - meanfunc.(xtest)) + logdet(K + Σ + 1e-6*diageye(length(y))) + N*log(2π))/2
end