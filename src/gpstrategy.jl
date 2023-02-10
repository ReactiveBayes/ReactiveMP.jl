export AbstractCovarianceStrategyType
export CovarianceMatrixStrategy
export FullCovarianceStrategy
export DeterministicInducingConditional, DIC, SoR, SubsetOfRegressors
export DeterministicTrainingConditional, DTC
export FullyIndependentTrainingConditional, FITC

import KernelFunctions: kernelmatrix, kernelmatrix!,Kernel 
abstract type AbstractCovarianceStrategyType end 


#------- Cache  ------- #
#use this cache to store and modify matrices 
struct GPCache
    cache_matrices::Dict{Tuple{Symbol, Tuple{Int, Int}}, Matrix{Float64}}
    cache_vectors::Dict{Tuple{Symbol, Int}, Vector{Float64}}
end

GPCache() = GPCache(Dict{Tuple{Symbol, Tuple{Int, Int}}, Matrix{Float64}}(), Dict{Tuple{Symbol, Int}, Vector{Float64}}())

function getcache(cache::GPCache, label::Tuple{Symbol, Tuple{Int, Int}})
    return get!(() -> Matrix{Float64}(undef, label[2]), cache.cache_matrices, label)
end

function getcache(cache::GPCache, label::Tuple{Symbol, Int})
    return get!(() -> Vector{Float64}(undef, label[2]), cache.cache_vectors, label)
end

#create kernelmatrix in-place
function kernelmatrix!(cache::GPCache, label::Symbol, kernel, x, y) 
    result = getcache(cache, (label, (length(x), length(y))))
    return kernelmatrix!(result, kernel, x, y)
end

function mul_A_B_At!(cache::GPCache, A, B)
    AB = getcache(cache, (:mul_A_B_At!_AB, (size(A, 1), size(B, 2))))
    ABAt = getcache(cache, (:mul_A_B_At!_ABAt, (size(AB, 1), size(A, 1))))
    mul!(AB, A, B)
    mul!(ABAt, AB, A')

    return ABAt
end
#------- Cache  ------- #

struct CovarianceMatrixStrategy{S} <: AbstractCovarianceStrategyType
    strategy :: S
end
CovarianceMatrixStrategy() = CovarianceMatrixStrategy(nothing)

#------- Full covariance matrix  ------- #

mutable struct FullCovarianceStrategy{N,R}
    n_inducing :: N 
    rng :: R 
    invKff :: AbstractArray   #store inverse of Kff  
    cache :: GPCache
end
FullCovarianceStrategy() = FullCovarianceStrategy(Int[],nothing,Float64[1;;], GPCache()) 

#------------- SoR ---------------#
mutable struct DeterministicInducingConditional{R}
    n_inducing :: Int 
    rng :: R 
    Kuu :: AbstractArray 
    Kuf :: AbstractArray 
    Σ   :: AbstractArray
    invΛ :: AbstractArray
    cache :: GPCache
end 
# mutable struct DeterministicInducingConditional{R}
#     n_inducing :: Int 
#     rng :: R 
#     cache :: GPCache 
# end
const DIC = DeterministicInducingConditional
const SoR = DeterministicInducingConditional
const SubsetOfRegressors = DeterministicInducingConditional

DeterministicInducingConditional(n_inducing) = DeterministicInducingConditional(n_inducing, MersenneTwister(1), Float64[1;;],Float64[1;;],Float64[1;;],Float64[1;;],GPCache())
# DeterministicInducingConditional(n_inducing) = DeterministicInducingConditional(n_inducing, MersenneTwister(1), GPCache())

#-------------- DTC -----------------#
mutable struct DeterministicTrainingConditional{R}
    n_inducing :: Int
    rng        :: R
    Kuu        :: AbstractArray 
    Kuf        :: AbstractArray
    Σ          :: AbstractArray
    invKuu     :: AbstractArray
    invΛ       :: AbstractArray
    cache      :: GPCache
end

const DTC = DeterministicTrainingConditional

DeterministicTrainingConditional(n_inducing) = DeterministicTrainingConditional(n_inducing, MersenneTwister(1), Float64[1;;],Float64[1;;],Float64[1;;],Float64[1;;],Float64[1;;],GPCache())

# -------------- FITC ----------------- #
mutable struct FullyIndependentTrainingConditional{R}
    n_inducing :: Int 
    rng        :: R 
    Kuu        :: AbstractArray
    Kuf        :: AbstractArray 
    Σ          :: AbstractArray
    invKuu     :: AbstractArray 
    invΛ       :: AbstractArray
    cache      :: GPCache
end 
const FITC = FullyIndependentTrainingConditional

FullyIndependentTrainingConditional(n_inducing) = FullyIndependentTrainingConditional(n_inducing, MersenneTwister(1),Float64[1;;],Float64[1;;],Float64[1;;],Float64[1;;],Float64[1;;],GPCache())

#--------------- GP prediction ------------------#
function predictMVN end 

predictMVN(gpstrategy::CovarianceMatrixStrategy{<:FullCovarianceStrategy}, kernelfunc, meanfunc, xtrain, xtest, y, inducing)                 = fullcov(gpstrategy,kernelfunc, meanfunc, xtrain, xtest, y)
predictMVN(gpstrategy::CovarianceMatrixStrategy{<:DeterministicInducingConditional}, kernelfunc, meanfunc, xtrain, xtest, y, inducing)       = sor(gpstrategy,kernelfunc, meanfunc, xtrain, xtest, y, inducing)
predictMVN(gpstrategy::CovarianceMatrixStrategy{<:DeterministicTrainingConditional}, kernelfunc, meanfunc, xtrain, xtest, y, inducing)       = dtc(gpstrategy,kernelfunc, meanfunc, xtrain, xtest, y, inducing)
predictMVN(gpstrategy::CovarianceMatrixStrategy{<:FullyIndependentTrainingConditional}, kernelfunc, meanfunc, xtrain, xtest, y, inducing)    = fitc(gpstrategy,kernelfunc, meanfunc, xtrain, xtest, y, inducing)

#### Full covariance strategy 
function fullcov(gpstrategy::CovarianceMatrixStrategy{<:FullCovarianceStrategy},kernelfunc, meanfunc,xtrain,xtest,y)  
    cache = gpstrategy.strategy.cache 
    Kfy                = kernelmatrix!(cache, :Kfy, kernelfunc,xtest,xtrain) #K*f
    Kff                = kernelmatrix!(cache, :Kff, kernelfunc,xtest,xtest)  #K**
    invKtrain          = length(gpstrategy.strategy.invKff)>1 ? gpstrategy.strategy.invKff : cholinv(kernelmatrix(kernelfunc,xtrain,xtrain) + 1e-6*I)  

    xtest_transformed = getcache(cache, (:xtest,length(xtest)))
    xtrain_transformed = getcache(cache, (:xtrain, length(xtrain)))
    map!(meanfunc,xtest_transformed, xtest)
    map!((z,x) -> z - meanfunc(x), xtrain_transformed, y, xtrain)

    result1 = getcache(cache, (:result1, size(invKtrain,1)))
    result2 = getcache(cache, (:result2, size(Kfy, 1)))

    mul!(result1, invKtrain, xtrain_transformed)
    mul!(result2, Kfy, result1)

    μ                  = xtest_transformed + result2 
    Σ                  = Kff - mul_A_B_At!(cache, Kfy, invKtrain) 
    return μ, Σ
end

#### SoR strategy 
function sor(gpstrategy::CovarianceMatrixStrategy{<:DeterministicInducingConditional}, kernelfunc, meanfunc, xtrain, xtest, y, x_u)
    cache = gpstrategy.strategy.cache 
    Kfu = kernelmatrix!(cache, :Kfu, kernelfunc,xtest,x_u) # cross covariance K_*u between test and inducing points 
    Σ = gpstrategy.strategy.Σ
    Kutrain = gpstrategy.strategy.Kuf
    invΛ = gpstrategy.strategy.invΛ

    xtrain_transformed = getcache(cache, (:xtrain, length(xtrain)))
    xtest_transformed = getcache(cache, (:xtest, length(xtest)))

    map!(meanfunc, xtest_transformed, xtest) # xtest = meanfunc.(xtest)
    map!((y,x) -> y - meanfunc(x), xtrain_transformed, y, xtrain) # xtrain_transformed = y .- meanfunc(xtrain)

    result1 = getcache(cache, (:result1, size(invΛ,1)))
    result2 = getcache(cache, (:result2, size(Kutrain,1)))
    result3 = getcache(cache, (:result3, size(Σ,1)))
    result4 = getcache(cache, (:result4, size(Kfu,1)))

    mul!(result1, invΛ, xtrain_transformed)
    mul!(result2, Kutrain, result1)
    mul!(result3, Σ, result2)
    mul!(result4, Kfu, result3)

    μ_SOR = xtest_transformed + result4
    Σ_SOR = mul_A_B_At!(cache, Kfu, Σ) 

    return μ_SOR, Σ_SOR
end

#### DTC strategy 
function dtc(gpstrategy::CovarianceMatrixStrategy{<:DeterministicTrainingConditional},kernelfunc, meanfunc, xtrain, xtest, y, x_u)

    cache = gpstrategy.strategy.cache

    Kfu = kernelmatrix!(cache, :Kfu, kernelfunc, xtest, x_u) # cross covariance K_*u between test and inducing points
    Kff = kernelmatrix!(cache, :Kff, kernelfunc, xtest, xtest) # K** the covariance of the test points  

    invΛ = gpstrategy.strategy.invΛ
    Kuf = gpstrategy.strategy.Kuf
    Σ = gpstrategy.strategy.Σ

    xtest_transformed = getcache(cache, (:xtest, length(xtest)))
    xtrain_transformed = getcache(cache, (:xtrain, length(xtrain)))

    map!(meanfunc, xtest_transformed, xtest) # xtest_transformed = meanfunc.(xtest)
    map!((y, x) -> y - meanfunc(x), xtrain_transformed, y, xtrain) # xtrain_transformed = y .- meanfunc.(xtrain)

    result1 = getcache(cache, (:result1, size(invΛ, 1)))
    result2 = getcache(cache, (:result2, size(Kuf, 1)))
    result3 = getcache(cache, (:result3, size(Σ, 1)))
    result4 = getcache(cache, (:result4, size(Kfu, 1)))

    mul!(result1, invΛ, xtrain_transformed)
    mul!(result2, Kuf, result1)
    mul!(result3, Σ, result2)
    mul!(result4, Kfu, result3)

    μ_DTC = xtest_transformed + result4

    Σ_DTC = Kff - mul_A_B_At!(cache, Kfu, gpstrategy.strategy.invKuu) 
    Σ_DTC = Σ_DTC + mul_A_B_At!(cache, Kfu, gpstrategy.strategy.Σ)
    return μ_DTC, Σ_DTC 
end

#### FITC strategy 
function fitc(gpstrategy::CovarianceMatrixStrategy{<:FullyIndependentTrainingConditional},kernelfunc, meanfunc, xtrain, xtest, y, x_u)
    cache = gpstrategy.strategy.cache 
    Kfu = kernelmatrix!(cache, :Kfu, kernelfunc,xtest,x_u) # cross covariance K_*u between test and inducing points
    Kff = kernelmatrix!(cache, :Kff, kernelfunc,xtest,xtest) # K** the covariance of the test points 
    #get information from strategy 
    Σ = gpstrategy.strategy.Σ
    Kutrain = gpstrategy.strategy.Kuf 
    invΛ = gpstrategy.strategy.invΛ
    invKuu = gpstrategy.strategy.invKuu 
    ## computation 
    xtest_transformed = getcache(cache, (:xtest, length(xtest)))
    xtrain_transformed = getcache(cache, (:xtrain, length(xtrain)))
    
    map!(meanfunc, xtest_transformed, xtest) #compute meanfunc.(xtest) and store in xtest_transformed
    map!((y,x) -> y - meanfunc(x), xtrain_transformed, y, xtrain) #compute meanfunc.(xtrain) and store in xtrain_transformed

    result1 = getcache(cache, (:result1, size(invΛ, 1)))
    result2 = getcache(cache, (:result2, size(Kutrain, 1)))
    result3 = getcache(cache, (:result3, size(Σ, 1)))
    result4 = getcache(cache, (:result4, size(Kfu, 1)))
    mul!(result1, invΛ, xtrain_transformed)
    mul!(result2, Kutrain, result1)
    mul!(result3, Σ, result2)
    mul!(result4, Kfu, result3)

    μ_FITC = xtest_transformed + result4

    Σ_FITC = Kff - mul_A_B_At!(cache, Kfu, invKuu)
    Σ_FITC = Diagonal(Σ_FITC) + mul_A_B_At!(cache,Kfu,Σ)

    return μ_FITC, Σ_FITC  
end

### function for extracting the matrices for strategies 
function extracmatrix! end 
function extractmatrix_change! end 
#--------------- full covariance strategy --------------------#
function extractmatrix!(gpstrategy::CovarianceMatrixStrategy{<:FullCovarianceStrategy}, kernel, x_train, Σ_noise, x_induc) 
    Kff = kernelmatrix(kernel,x_train,x_train)
    gpstrategy.strategy.invKff = cholinv(Kff + Diagonal(Σ_noise))

    return gpstrategy
end

function extractmatrix_change!(gpstrategy::CovarianceMatrixStrategy{<:FullCovarianceStrategy}, kernel, x_train, Σ_noise, x_induc) 
    return extractmatrix!(gpstrategy,kernel, x_train, Σ_noise,x_induc)
end
#--------------- SoR strategy -------------------------#
function extractmatrix!(gpstrategy::CovarianceMatrixStrategy{<:DeterministicInducingConditional}, kernel, x_train, Σ_noise, x_induc)
    gpcache = gpstrategy.strategy.cache 
    Kuu = kernelmatrix(kernel,x_induc,x_induc) 
    Kuf = kernelmatrix(kernel,x_induc,x_train) #cross covariance between inducing and training points 
    invΛ = cholinv(Diagonal(Σ_noise))

    Σ = Kuu + mul_A_B_At!(gpcache, Kuf, invΛ)
    Σ = cholinv(Σ) 

    gpstrategy.strategy.Kuf = Kuf 
    gpstrategy.strategy.Σ = Σ
    gpstrategy.strategy.invΛ = invΛ
    gpstrategy.strategy.Kuu = Kuu
    return gpstrategy
end

function extractmatrix_change!(gpstrategy::CovarianceMatrixStrategy{<:DeterministicInducingConditional}, kernel, x_train, Σ_noise, x_induc)
    gpcache = gpstrategy.strategy.cache 
    Kuu = gpstrategy.strategy.Kuu 
    Kuf = kernelmatrix(kernel,x_induc,x_train) #cross covariance between inducing and training points 
    invΛ = cholinv(Diagonal(Σ_noise))
    Σ = Kuu + mul_A_B_At!(gpcache, Kuf, invΛ)
    Σ = cholinv(Σ)

    gpstrategy.strategy.Kuf = Kuf 
    gpstrategy.strategy.Σ = Σ
    gpstrategy.strategy.invΛ = invΛ
    return gpstrategy
end
#--------------- DTC strategy -------------------------#
function extractmatrix!(gpstrategy::CovarianceMatrixStrategy{<:DeterministicTrainingConditional}, kernel, x_train, Σ_noise, x_induc)
    gpcache = gpstrategy.strategy.cache 
    Kuf = kernelmatrix(kernel,x_induc,x_train) #cross covariance between inducing and training points 
    Kuu = kernelmatrix(kernel,x_induc,x_induc) 
    invΛ = cholinv(Diagonal(Σ_noise))

    Σ = Kuu + mul_A_B_At!(gpcache, Kuf, invΛ)
    Σ = cholinv(Σ) 

    gpstrategy.strategy.Kuf = Kuf 
    gpstrategy.strategy.Σ = Σ
    gpstrategy.strategy.invKuu = cholinv(Kuu)
    gpstrategy.strategy.invΛ = invΛ
    gpstrategy.strategy.Kuu = Kuu
    return gpstrategy
end

function extractmatrix_change!(gpstrategy::CovarianceMatrixStrategy{<:DeterministicTrainingConditional}, kernel, x_train, Σ_noise, x_induc)
    gpcache = gpstrategy.strategy.cache 
    Kuf = kernelmatrix(kernel,x_induc,x_train) #cross covariance between inducing and training points 
    Kuu = gpstrategy.strategy.Kuu 

    invΛ = cholinv(Diagonal(Σ_noise))
    Σ = Kuu + mul_A_B_At!(gpcache, Kuf, invΛ)
    Σ = cholinv(Σ)

    gpstrategy.strategy.Kuf = Kuf 
    gpstrategy.strategy.Σ = Σ
    gpstrategy.strategy.invΛ = invΛ
end

#-------------- FITC strategy --------------------------------#
function extractmatrix!(gpstrategy::CovarianceMatrixStrategy{<:FullyIndependentTrainingConditional}, kernel, x_train, Σ_noise, x_induc)
    gpcache = gpstrategy.strategy.cache 
    Kuu = kernelmatrix(kernel,x_induc,x_induc)
    Kuf = kernelmatrix(kernel, x_induc, x_train) # cross covariance between inducing and training points 
    Kff = kernelmatrix(kernel,x_train,x_train)

    invKuu = cholinv(Kuu)
    Λ = Kff - mul_A_B_At!(gpcache,Kuf',invKuu)
    Λ = Diagonal(Λ + Σ_noise)
    invΛ = cholinv(Λ)

    Σ = Kuu + mul_A_B_At!(gpcache,Kuf, invΛ)
    Σ = cholinv(Σ)

    gpstrategy.strategy.Kuu = Kuu 
    gpstrategy.strategy.Kuf =  Kuf 
    gpstrategy.strategy.Σ = Σ
    gpstrategy.strategy.invΛ = invΛ
    gpstrategy.strategy.invKuu = invKuu
    return gpstrategy
end 

function extractmatrix_change!(gpstrategy::CovarianceMatrixStrategy{<:FullyIndependentTrainingConditional}, kernel, x_train, Σ_noise, x_induc)
    gpcache = gpstrategy.strategy.cache 
    Kuf = kernelmatrix(kernel, x_induc, x_train) # cross covariance between inducing and training points 
    Kff = kernelmatrix(kernel,x_train,x_train)
    invKuu = gpstrategy.strategy.invKuu
    Kuu = gpstrategy.strategy.Kuu

    Λ = Kff - mul_A_B_At!(gpcache,Kuf',invKuu)
    Λ = Diagonal(Λ + Σ_noise)
    invΛ = cholinv(Λ)

    Σ = Kuu + mul_A_B_At!(gpcache,Kuf, invΛ)
    Σ = cholinv(Σ)

    gpstrategy.strategy.Kuf =  Kuf 
    gpstrategy.strategy.Σ = Σ
    gpstrategy.strategy.invΛ = invΛ
return gpstrategy
end
