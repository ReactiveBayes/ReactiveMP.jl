export GaussianProcess, ProcessMeta 

# Create GP structure 
struct GaussianProcess{F, K, M, I1, I2, I3, I4, I5, S}
    meanfunction        ::F
    kernelfunction      ::K
    finitemarginal      ::M  
    testinput           ::I1
    traininput          ::I2
    observation         ::I3
    observation_noise   ::I4
    inducing_input      ::I5
    covariance_strategy ::S
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
    return ((y - meanfunc.(xtest))' * cholinv(Σ) * (y - meanfunc.(xtest)) + logdet(Σ + 1e-6*diageye(length(y))) + N*log(2π))/2
end

function Distributions.entropy(pm::PointMass{F}) where {F <: Function}
    return ReactiveMP.CountingReal(Float64,-1)
end

function Distributions.entropy(pm::PointMass{F}) where {F <: Kernel}
    return ReactiveMP.CountingReal(Float64,-1)
end

function ReactiveMP.entropy(p::GaussianProcess)
    return ReactiveMP.entropy(p.finitemarginal)
end