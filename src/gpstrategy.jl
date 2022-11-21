export AbstractCovarianceStrategyType
export CovarianceMatrixStrategy
export FullCovarianceStrategy
export DeterministicInducingConditional, DIC, SoR, SubsetOfRegressors
export DeterministicTrainingConditional, DTC
export FullyIndependentTrainingConditional, FITC

import KernelFunctions: kernelmatrix, Kernel 
abstract type AbstractCovarianceStrategyType end 

struct CovarianceMatrixStrategy{S} <: AbstractCovarianceStrategyType
    strategy :: S
end
CovarianceMatrixStrategy() = CovarianceMatrixStrategy(nothing)

#------- Full covariance matrix  ------- #

mutable struct FullCovarianceStrategy
    n_inducing
    rng
    invKff    #store inverse of Kff  
end
FullCovarianceStrategy() = FullCovarianceStrategy(nothing,nothing,nothing) 

#------------- SoR ---------------#
mutable struct DeterministicInducingConditional
    n_inducing
    rng
    Kuu
    Kuf 
    Σ
    invΛ
end 
const DIC = DeterministicInducingConditional
const SoR = DeterministicInducingConditional
const SubsetOfRegressors = DeterministicInducingConditional

DeterministicInducingConditional(n_inducing) = DeterministicInducingConditional(n_inducing, MersenneTwister(1),nothing,nothing,nothing,nothing)

#-------------- DTC -----------------#
mutable struct DeterministicTrainingConditional
    n_inducing
    rng 
    Kuu
    Kuf
    Σ
    invKuu 
    invΛ
end
const DTC = DeterministicTrainingConditional

DeterministicTrainingConditional(n_inducing) = DeterministicTrainingConditional(n_inducing, MersenneTwister(1),nothing,nothing,nothing,nothing,nothing)

# -------------- FITC ----------------- #
mutable struct FullyIndependentTrainingConditional
    n_inducing
    rng 
    Kuu
    Kuf
    Σ
    invKuu 
    invΛ
end 
const FITC = FullyIndependentTrainingConditional

FullyIndependentTrainingConditional(n_inducing) = FullyIndependentTrainingConditional(n_inducing, MersenneTwister(1),nothing,nothing,nothing,nothing,nothing)

#--------------- GP prediction ------------------#
function predictMVN end 

predictMVN(gpstrategy::CovarianceMatrixStrategy{FullCovarianceStrategy}, kernelfunc, meanfunc, xtrain, xtest, y, inducing::Nothing)                       = fullcov(gpstrategy,kernelfunc, meanfunc, xtrain, xtest, y)
predictMVN(gpstrategy::CovarianceMatrixStrategy{DeterministicInducingConditional}, kernelfunc,meanfunc, xtrain, xtest, y, inducing::AbstractArray)        = sor(gpstrategy,kernelfunc, meanfunc, xtrain, xtest, y, inducing)
predictMVN(gpstrategy::CovarianceMatrixStrategy{DeterministicTrainingConditional},kernelfunc, meanfunc, xtrain, xtest, y, inducing::AbstractArray)        = dtc(gpstrategy,kernelfunc, meanfunc, xtrain, xtest, y, inducing)
predictMVN(gpstrategy::CovarianceMatrixStrategy{FullyIndependentTrainingConditional}, kernelfunc, meanfunc, xtrain, xtest, y, inducing::AbstractArray)    = fitc(gpstrategy,kernelfunc, meanfunc, xtrain, xtest, y, inducing)

#### Full covariance strategy 
function fullcov(gpstrategy,kernelfunc,meanfunc,xtrain,xtest,y)  # function for computing GP marginal 
    Kfy                = kernelmatrix(kernelfunc,xtest,xtrain) #K*f
    Kff                = kernelmatrix(kernelfunc,xtest,xtest)  #K**

    μ                  = meanfunc.(xtest) + Kfy*gpstrategy.strategy.invKff *(y-meanfunc.(xtrain)) 
    Σ                  = Kff - Kfy*gpstrategy.strategy.invKff *Kfy'
    return μ, Σ
end

#### SoR strategy 
function sor(gpstrategy,kernelfunc,meanfunc, xtrain, xtest, y, x_u)
    Kfu = kernelmatrix(kernelfunc,xtest,x_u) # cross covariance K_*u between test and inducing points 

    μ_SOR = meanfunc.(xtest) + Kfu * gpstrategy.strategy.Σ * gpstrategy.strategy.Kuf * gpstrategy.strategy.invΛ * (y - meanfunc.(xtrain))
    Σ_SOR = Kfu * gpstrategy.strategy.Σ * Kfu'

    return μ_SOR, Σ_SOR
end

#### DTC strategy 
function dtc(gpstrategy,kernelfunc, meanfunc, xtrain, xtest, y, x_u)
    Kfu = kernelmatrix(kernelfunc,xtest,x_u) # cross covariance K_*u between test and inducing points
    Kff = kernelmatrix(kernelfunc,xtest,xtest) # K** the covariance of the test points  

    μ_DTC = meanfunc.(xtest) + Kfu * gpstrategy.strategy.Σ * gpstrategy.strategy.Kuf * gpstrategy.strategy.invΛ * (y - meanfunc.(xtrain))
    Σ_DTC = Kff - Kfu * gpstrategy.strategy.invKuu * Kfu' + Kfu * gpstrategy.strategy.Σ * Kfu'

    return μ_DTC, Σ_DTC 
end

#### FITC strategy 
function fitc(gpstrategy,kernelfunc, meanfunc, xtrain, xtest, y, x_u)
    Kfu = kernelmatrix(kernelfunc,xtest,x_u) # this is K*u
    Kff = kernelmatrix(kernelfunc,xtest,xtest) # this is K**

    μ_FITC = meanfunc.(xtest) + Kfu * gpstrategy.strategy.Σ * gpstrategy.strategy.Kuf * gpstrategy.strategy.invΛ * (y - meanfunc.(xtrain))
    Σ_FITC = Diagonal(Kff - Kfu * gpstrategy.strategy.invKuu  * Kfu')  + Kfu * gpstrategy.strategy.Σ * Kfu'

    return μ_FITC, Σ_FITC  
end

### function for extracting the matrices for strategies 
function extracmatrix! end 
#--------------- full covariance strategy --------------------#
function extractmatrix!(gpstrategy::CovarianceMatrixStrategy{FullCovarianceStrategy}, kernel::Kernel, x_train::AbstractArray, Σ_noise::AbstractArray , x_induc::Nothing, s::Bool) 
    Kff = kernelmatrix(kernel,x_train,x_train)
    gpstrategy.strategy.invKff = cholinv(Kff + Diagonal(Σ_noise))

    return gpstrategy
end

#--------------- SoR strategy -------------------------#
function extractmatrix!(gpstrategy::CovarianceMatrixStrategy{DeterministicInducingConditional}, kernel::Kernel, x_train::AbstractArray, Σ_noise::AbstractArray, x_induc::AbstractArray, s::Bool)
    if s
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
function extractmatrix!(gpstrategy::CovarianceMatrixStrategy{DeterministicTrainingConditional}, kernel::Kernel, x_train::AbstractArray, Σ_noise::AbstractArray, x_induc::AbstractArray, s::Bool)
    if s
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
function extractmatrix!(gpstrategy::CovarianceMatrixStrategy{FullyIndependentTrainingConditional}, kernel::Kernel, x_train::AbstractArray, Σ_noise::AbstractArray, x_induc::AbstractArray, s::Bool)
    if s
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

