export GaussianProcess, ProcessMeta 

# Create GP structure 
struct GaussianProcess
    meanfunction        ::Function 
    kernelfunction      ::Kernel
    finitemarginal      
    testinput           ::Array{Float64}    
    traininput          ::Array{Float64}
    observation         ::Array{Float64}
    observation_noise   ::Array{Float64}     
    inducing_input      ::Array{Float64}
    covariance_strategy ::AbstractCovarianceStrategyType
end

struct ProcessMeta 
    index :: Int 
end 

@node GaussianProcess Stochastic [out, meanfunc, kernelfunc, params]

### compute free energy here 