export AbstractCovarianceStrategyType
export CovarianceMatrixStrategy
export FullCovarianceStrategy
export DeterministicInducingConditional, DIC, SoR, SubsetOfRegressors
export DeterministicTrainingConditional, DTC
export FullyIndependentTrainingConditional, FITC

import KernelFunctions: kernelmatrix 
abstract type AbstractCovarianceStrategyType end 

struct CovarianceMatrixStrategy{S} <: AbstractCovarianceStrategyType
    strategy :: S
end
CovarianceMatrixStrategy() = CovarianceMatrixStrategy(nothing)

struct FullCovarianceStrategy
    n_inducing
    rng
end
FullCovarianceStrategy() = FullCovarianceStrategy(nothing,nothing) 
#############
struct DeterministicInducingConditional
    n_inducing
    rng 
end 
const DIC = DeterministicInducingConditional
const SoR = DeterministicInducingConditional
const SubsetOfRegressors = DeterministicInducingConditional

DeterministicInducingConditional(n_inducing) = DeterministicInducingConditional(n_inducing, MersenneTwister(1))
#################
struct DeterministicTrainingConditional
    n_inducing
    rng 
end
const DTC = DeterministicTrainingConditional

DeterministicTrainingConditional(n_inducing) = DeterministicTrainingConditional(n_inducing, MersenneTwister(1))
###############
struct FullyIndependentTrainingConditional
    n_inducing
    rng 
end 
const FITC = FullyIndependentTrainingConditional

FullyIndependentTrainingConditional(n_inducing) = FullyIndependentTrainingConditional(n_inducing, MersenneTwister(1))
####################
function predictMVN end 

predictMVN(::CovarianceMatrixStrategy{FullCovarianceStrategy}, kernelfunc, meanfunc, xtrain, xtest, y, Σy, inducing::Nothing)                       = fullcov(kernelfunc, meanfunc, xtrain, xtest, y, Σy)
predictMVN(::CovarianceMatrixStrategy{DeterministicInducingConditional}, kernelfunc,meanfunc, xtrain, xtest, y, Σy, inducing::AbstractArray)        = sor(kernelfunc, meanfunc, xtrain, xtest, y, Σy, inducing)
predictMVN(::CovarianceMatrixStrategy{DeterministicTrainingConditional},kernelfunc, meanfunc, xtrain, xtest, y, Σy, inducing::AbstractArray)        = dtc(kernelfunc, meanfunc, xtrain, xtest, y, Σy, inducing)
predictMVN(::CovarianceMatrixStrategy{FullyIndependentTrainingConditional}, kernelfunc, meanfunc, xtrain, xtest, y, Σy, inducing::AbstractArray)    = fitc(kernelfunc, meanfunc, xtrain, xtest, y, Σy, inducing)

#### Full covariance strategy 
function fullcov(kernelfunc,meanfunc,xtrain,xtest,y,Σy)  # function for computing GP marginal 
    Kyy                = kernelmatrix(kernelfunc,xtrain,xtrain)
    Kfy                = kernelmatrix(kernelfunc,xtest,xtrain)
    Kff                = kernelmatrix(kernelfunc,xtest,xtest)
    μ                  = meanfunc.(xtest) + Kfy*cholinv(Kyy+Σy)*(y-meanfunc.(xtrain)) 
    Σ                  = Kff - Kfy*cholinv(Kyy+Σy)*Kfy'    
    return μ, Σ
end

#### SoR strategy 
function sor(kernelfunc,meanfunc, xtrain, xtest, y, Σy, x_u)
    """
    x_u: inducing point 
    """
    Kuu = kernelmatrix(kernelfunc,x_u,x_u)
    invKuu = cholinv(Kuu)
    Kyu = kernelmatrix(kernelfunc,xtrain,x_u)
    Kfu = kernelmatrix(kernelfunc,xtest,x_u)

    ## calculate Q's matrices
    Qyy = Kyu * invKuu * Kyu' # approximate covariance matrix for training set  
    Qff = Kfu * invKuu * Kfu' # approximate covariance matrix for test set 
    Qfy = Kfu * invKuu * Kyu' # approximate cross covariance matrix 

    μ_DIC = meanfunc.(xtest) + Qfy * cholinv(Qyy + Σy) * (y - meanfunc.(xtrain))
    Σ_DIC = Qff - Qfy * cholinv(Qyy + Σy) * Qfy'

    return μ_DIC, Σ_DIC
end

#### DTC strategy 
function dtc(kernelfunc, meanfunc, xtrain, xtest, y, Σy, x_u)
    """
    x_u: inducing point 
    """
    Kuu = kernelmatrix(kernelfunc,x_u,x_u)
    invKuu = cholinv(Kuu)
    Kyu = kernelmatrix(kernelfunc,xtrain,x_u)
    Kfu = kernelmatrix(kernelfunc,xtest,x_u)
    Kff = kernelmatrix(kernelfunc,xtest,xtest)

    # Calculate Q's matrices 
    Qyy = Kyu * invKuu * Kyu' # approximate covariance matrix for training set  
    Qfy = Kfu * invKuu * Kyu' # approximate cross covariance matrix 

    μ_DTC = meanfunc.(xtest) + Qfy * cholinv(Qyy + Σy) * (y - meanfunc.(xtrain))
    Σ_DTC = Kff - Qfy * cholinv(Qyy + Σy) * Qfy'

    return μ_DTC, Σ_DTC 
end

#### FITC strategy 
function fitc(kernelfunc, meanfunc, xtrain, xtest, y, Σy, x_u)
    """
    x_u : inducing point 
    """
    Kuu = kernelmatrix(kernelfunc,x_u,x_u)
    invKuu = cholinv(Kuu)
    Kyu = kernelmatrix(kernelfunc,xtrain,x_u)
    Kfu = kernelmatrix(kernelfunc,xtest,x_u)
    Kff = kernelmatrix(kernelfunc,xtest,xtest)
    Kyy = kernelmatrix(kernelfunc,xtrain,xtrain)

    # Calculate Q's matrices 
    Qyy = Kyu * invKuu * Kyu' # approximate covariance matrix for training set  
    Qfy = Kfu * invKuu * Kyu' # approximate cross covariance matrix 

    Λ = Diagonal(Kyy - Qyy + Σy)

    μ_FITC = meanfunc.(xtest) + Qfy * cholinv(Qyy + Λ) * (y - meanfunc.(xtrain))
    Σ_FITC = Kff - Qfy * cholinv(Qyy + Λ) * Qfy'

    return μ_FITC, Σ_FITC  
end
