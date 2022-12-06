export AbstractCovarianceStrategyType
export CovarianceMatrixStrategy
export FullCovarianceStrategy
export DeterministicInducingConditional, DIC, SoR, SubsetOfRegressors
export DeterministicTrainingConditional, DTC
export FullyIndependentTrainingConditional, FITC

import KernelFunctions: kernelmatrix, kernelmatrix!,Kernel 
abstract type AbstractCovarianceStrategyType end 

struct CovarianceMatrixStrategy{S} <: AbstractCovarianceStrategyType
    strategy :: S
end
CovarianceMatrixStrategy() = CovarianceMatrixStrategy(nothing)

#------- Full covariance matrix  ------- #

mutable struct FullCovarianceStrategy{N,R}
    n_inducing :: N 
    rng :: R 
    Kff :: Array{Float64}
    invKff :: Array{Float64}    #store inverse of Kff  
end
FullCovarianceStrategy() = FullCovarianceStrategy(Int[],nothing,Float64[1;;], Float64[1;;]) 

#------------- SoR ---------------#
mutable struct DeterministicInducingConditional{R}
    n_inducing :: Int 
    rng :: R 
    Kuu :: Array{Float64} 
    Kuf :: Array{Float64} 
    Σ   :: AbstractArray
    invΛ :: AbstractArray
end 
const DIC = DeterministicInducingConditional
const SoR = DeterministicInducingConditional
const SubsetOfRegressors = DeterministicInducingConditional

DeterministicInducingConditional(n_inducing) = DeterministicInducingConditional(n_inducing, MersenneTwister(1), Float64[1;;],Float64[1;;],Float64[1;;],Float64[1;;])

#-------------- DTC -----------------#
mutable struct DeterministicTrainingConditional{R}
    n_inducing :: Int
    rng        :: R
    Kuu        :: Array{Float64}   
    Kuf        :: AbstractArray  # this is a vector of matrices. It stores 2 matrices.
    Σ          :: AbstractArray    
    invKuu     :: Array{Float64} 
    invΛ       :: AbstractArray
end
const DTC = DeterministicTrainingConditional

DeterministicTrainingConditional(n_inducing) = DeterministicTrainingConditional(n_inducing, MersenneTwister(1),Float64[1;;],Float64[1.;;],Float64[1;;],Float64[1;;], Float64[1.;;])

# -------------- FITC ----------------- #
mutable struct FullyIndependentTrainingConditional{R}
    n_inducing :: Int 
    rng        :: R 
    Kuu        :: Array{Float64} 
    Kuf        :: Array{Float64} 
    Σ          :: AbstractArray
    invKuu     :: Array{Float64} 
    invΛ       :: AbstractArray
    Kff        :: Array{Float64}
end 
const FITC = FullyIndependentTrainingConditional

FullyIndependentTrainingConditional(n_inducing) = FullyIndependentTrainingConditional(n_inducing, MersenneTwister(1),Float64[1;;],Float64[1;;],Float64[1;;],Float64[1;;], Float64[1;;],Float64[1;;])

#--------------- GP prediction ------------------#
function predictMVN end 

predictMVN(gpstrategy::CovarianceMatrixStrategy{<:FullCovarianceStrategy}, kernelfunc::Kernel, meanfunc::Function, xtrain::Array{Float64}, xtest::Array{Float64}, y::Array{Float64}, inducing::Array{Float64})                 = fullcov(gpstrategy,kernelfunc, meanfunc, xtrain, xtest, y)
predictMVN(gpstrategy::CovarianceMatrixStrategy{<:DeterministicInducingConditional}, kernelfunc::Kernel, meanfunc::Function, xtrain::Array{Float64}, xtest::Array{Float64}, y::Array{Float64}, inducing::Array{Float64})       = sor(gpstrategy,kernelfunc, meanfunc, xtrain, xtest, y, inducing)
predictMVN(gpstrategy::CovarianceMatrixStrategy{<:DeterministicTrainingConditional}, kernelfunc::Kernel, meanfunc::Function, xtrain::Array{Float64}, xtest::Array{Float64}, y::Array{Float64}, inducing::Array{Float64})       = dtc(gpstrategy,kernelfunc, meanfunc, xtrain, xtest, y, inducing)
predictMVN(gpstrategy::CovarianceMatrixStrategy{<:FullyIndependentTrainingConditional}, kernelfunc::Kernel, meanfunc::Function, xtrain::Array{Float64}, xtest::Array{Float64}, y::Array{Float64}, inducing::Array{Float64})    = fitc(gpstrategy,kernelfunc, meanfunc, xtrain, xtest, y, inducing)

#### Full covariance strategy 
function fullcov(gpstrategy::CovarianceMatrixStrategy{<:FullCovarianceStrategy},kernelfunc::Kernel,meanfunc::Function,xtrain::Array{Float64},xtest::Array{Float64},y::Array{Float64})  # function for computing GP marginal 
    Kfy                = kernelmatrix(kernelfunc,xtest,xtrain) #K*f
    Kff                = kernelmatrix(kernelfunc,xtest,xtest)  #K**
    copyto!(y, y - meanfunc.(xtrain))

    μ                  = meanfunc.(xtest) + Kfy*gpstrategy.strategy.invKff *(y-meanfunc.(xtrain)) 
    Σ                  = Kff - Kfy*gpstrategy.strategy.invKff *Kfy'
    return μ, Σ
end

#### SoR strategy 
function sor(gpstrategy::CovarianceMatrixStrategy{<:DeterministicInducingConditional}, kernelfunc::Kernel, meanfunc::Function, xtrain::Array{Float64}, xtest::Array{Float64}, y::Array{Float64}, x_u::Array{Float64})
    Kfu = kernelmatrix(kernelfunc,xtest,x_u) # cross covariance K_*u between test and inducing points 

    μ_SOR = meanfunc.(xtest) + Kfu * gpstrategy.strategy.Σ * gpstrategy.strategy.Kuf * gpstrategy.strategy.invΛ * (y - meanfunc.(xtrain))
    Σ_SOR = Kfu * gpstrategy.strategy.Σ * Kfu'

    return μ_SOR, Σ_SOR
end

#### DTC strategy 
function dtc(gpstrategy::CovarianceMatrixStrategy{<:DeterministicTrainingConditional},kernelfunc::Kernel, meanfunc::Function, xtrain::Array{Float64}, xtest::Array{Float64}, y::Array{Float64}, x_u::Array{Float64})
    Kfu = kernelmatrix(kernelfunc,xtest,x_u) # cross covariance K_*u between test and inducing points
    Kff = kernelmatrix(kernelfunc,xtest,xtest) # K** the covariance of the test points  

    μ_DTC = meanfunc.(xtest) + Kfu * gpstrategy.strategy.Σ * gpstrategy.strategy.Kuf * gpstrategy.strategy.invΛ * (y - meanfunc.(xtrain))
    Σ_DTC = Kff - Kfu * gpstrategy.strategy.invKuu * Kfu' + Kfu * gpstrategy.strategy.Σ * Kfu'
    return μ_DTC, Σ_DTC 
end

#### FITC strategy 
function fitc(gpstrategy::CovarianceMatrixStrategy{<:FullyIndependentTrainingConditional},kernelfunc::Kernel, meanfunc::Function, xtrain::Array{Float64}, xtest::Array{Float64}, y::Array{Float64}, x_u::Array{Float64})
    Kfu = kernelmatrix(kernelfunc,xtest,x_u) # this is K*u
    Kff = kernelmatrix(kernelfunc,xtest,xtest) # this is K**

    μ_FITC = meanfunc.(xtest) + Kfu * gpstrategy.strategy.Σ * gpstrategy.strategy.Kuf * gpstrategy.strategy.invΛ * (y - meanfunc.(xtrain))
    Σ_FITC = Diagonal(Kff - Kfu * gpstrategy.strategy.invKuu  * Kfu')  + Kfu * gpstrategy.strategy.Σ * Kfu'

    return μ_FITC, Σ_FITC  
end

### function for extracting the matrices for strategies 
function extracmatrix! end 
#--------------- full covariance strategy --------------------#
function extractmatrix!(gpstrategy::CovarianceMatrixStrategy{<:FullCovarianceStrategy}, kernel::Kernel, x_train::Array{Float64}, Σ_noise::AbstractArray, x_induc::Array{Float64}) 
    Kff = kernelmatrix(kernel,x_train,x_train)
    gpstrategy.strategy.invKff = cholinv(Kff + Diagonal(Σ_noise))

    return gpstrategy
end

#--------------- SoR strategy -------------------------#
function extractmatrix!(gpstrategy::CovarianceMatrixStrategy{<:DeterministicInducingConditional}, kernel::Kernel, x_train::Array{Float64}, Σ_noise::AbstractArray, x_induc::Array{Float64})
    if length(gpstrategy.strategy.Kuu) == 1
        Kuu = kernelmatrix(kernel,x_induc,x_induc) 
        Kuf = kernelmatrix(kernel,x_induc,x_train) #cross covariance between inducing and training points 
        invΛ = cholinv(Diagonal(Σ_noise))
        Σ = cholinv(Kuu + Kuf * invΛ * Kuf') 

        gpstrategy.strategy.Kuf = Kuf 
        gpstrategy.strategy.Σ = Σ
        gpstrategy.strategy.invΛ = invΛ
        gpstrategy.strategy.Kuu = Kuu
    else 
        Kuf = kernelmatrix(kernel,x_induc,x_train) #cross covariance between inducing and training points 
        invΛ = cholinv(Diagonal(Σ_noise))
        Σ = cholinv(gpstrategy.strategy.Kuu + Kuf * invΛ * Kuf') 

        gpstrategy.strategy.Kuf = Kuf 
        gpstrategy.strategy.Σ = Σ
        gpstrategy.strategy.invΛ = invΛ
    end
    return gpstrategy
end

#--------------- DTC strategy -------------------------#
function extractmatrix!(gpstrategy::CovarianceMatrixStrategy{<:DeterministicTrainingConditional}, kernel::Kernel, x_train::Array{Float64}, Σ_noise::AbstractArray, x_induc::Array{Float64})
    if length(gpstrategy.strategy.Kuu) == 1
        Kuu = kernelmatrix(kernel,x_induc,x_induc) 
        Kuf = kernelmatrix(kernel,x_induc,x_train) #cross covariance between inducing and training points 
        invΛ = cholinv(Diagonal(Σ_noise))
        Σ = cholinv(Kuu + Kuf * invΛ * Kuf') 

        gpstrategy.strategy.Kuf = Kuf 
        gpstrategy.strategy.Σ = Σ
        gpstrategy.strategy.invKuu = cholinv(Kuu)
        gpstrategy.strategy.invΛ = invΛ
        gpstrategy.strategy.Kuu = Kuu
    else 
        Kuf = kernelmatrix(kernel,x_induc,x_train) #cross covariance between inducing and training points 
        invΛ = cholinv(Diagonal(Σ_noise))
        Σ = cholinv(gpstrategy.strategy.Kuu + Kuf * invΛ * Kuf') 

        gpstrategy.strategy.Kuf = Kuf 
        gpstrategy.strategy.Σ = Σ
        gpstrategy.strategy.invΛ = invΛ
    end
    return gpstrategy
end


#-------------- FITC strategy --------------------------------#
function extractmatrix!(gpstrategy::CovarianceMatrixStrategy{<:FullyIndependentTrainingConditional}, kernel::Kernel, x_train::Array{Float64}, Σ_noise::AbstractArray, x_induc::Array{Float64})
    if length(gpstrategy.strategy.Kuu) == 1
        Kuu = kernelmatrix(kernel,x_induc,x_induc)
        Kuf = kernelmatrix(kernel, x_induc, x_train) # cross covariance between inducing and training points 
        Kff = kernelmatrix(kernel,x_train,x_train)

        invKuu = cholinv(Kuu)
        Λ = Diagonal(Kff - Kuf' * invKuu * Kuf + Σ_noise)
        invΛ = cholinv(Λ)
        Σ = cholinv(Kuu + Kuf * invΛ * Kuf')

        gpstrategy.strategy.Kuu = Kuu 
        gpstrategy.strategy.Kuf =  Kuf 
        gpstrategy.strategy.Σ = Σ
        gpstrategy.strategy.invΛ = invΛ
        gpstrategy.strategy.invKuu = invKuu
    else
        Kuf = kernelmatrix(kernel, x_induc, x_train) # cross covariance between inducing and training points 
        Kff = kernelmatrix(kernel,x_train,x_train)

        Λ = Diagonal(Kff - Kuf' * gpstrategy.strategy.invKuu * Kuf + Σ_noise)
        invΛ = cholinv(Λ)
        Σ = cholinv(gpstrategy.strategy.Kuu + Kuf * invΛ * Kuf')

        gpstrategy.strategy.Kuf =  Kuf 
        gpstrategy.strategy.Σ = Σ
        gpstrategy.strategy.invΛ = invΛ
    end
    return gpstrategy
end

# function cholinv!(A::Matrix, B::AbstractArray)
#     invB = cholinv(B)
#     copyto!(A,invB)
#     return A 
# end