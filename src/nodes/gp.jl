export GaussianProcess, ProcessMeta 

# Create GP structure 
struct GaussianProcess 
    meanfunction        :: Function 
    kernelfunction      :: Kernel
    finitemarginal      
    testinput                
    traininput
    observation 
    observation_noise          
    inducing_input      
    covariance_strategy 
end

struct ProcessMeta 
    index :: Int 
end 

@node GaussianProcess Stochastic [out, meanfunc, kernelfunc, params]

### compute free energy here 