export AbstractCovarianceStrategyType
export CovarianceMatrixStrategy
export FullCovarianceStrategy
export DeterministicInducingConditional, DIC, SoR, SubsetOfRegressors
export DeterministicTrainingConditional, DTC
export FullyIndependentTrainingConditional, FITC

using KernelFunctions

import KernelFunctions: kernelmatrix, kernelmatrix!,Kernel 
abstract type AbstractCovarianceStrategyType end 


#------- Cache  ------- #

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

function KernelFunctions.kernelmatrix!(cache::GPCache, label::Symbol, kernel, x, y)
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
    Kuu   
    Kuf   
    Σ     
    invKuu
    invΛ  
    cache      :: GPCache
end

const DTC = DeterministicTrainingConditional

DeterministicTrainingConditional(n_inducing) = DeterministicTrainingConditional(n_inducing, MersenneTwister(1), Float64[1;;],Float64[1;;],Float64[1;;],Float64[1;;],Float64[1;;],GPCache())

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
function extractmatrix!(gpstrategy::CovarianceMatrixStrategy{<:DeterministicTrainingConditional}, kernel::Kernel, x_train::AbstractArray, Σ_noise::AbstractArray, x_induc::AbstractArray)
    cache = gpstrategy.strategy.cache

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
        Kuu = gpstrategy.strategy.Kuu
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