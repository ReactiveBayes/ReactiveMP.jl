export AbstractCovarianceStrategyType
export CovarianceMatrixStrategy
export FullCovarianceStrategy
export CirculantFullCovarianceStrategy
export DeterministicInducingConditional, DIC, SoR, SubsetOfRegressors
export DeterministicTrainingConditional, DTC
export FullyIndependentTrainingConditional, FITC
export RandomFourierFeature, RFF

import KernelFunctions: kernelmatrix, kernelmatrix!,Kernel, ScaledKernel, TransformedKernel, SqExponentialKernel, Transform 
import ToeplitzMatrices: Circulant
import FFTW: plan_rfft, plan_irfft 
import StableRNGs: StableRNG

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

mutable struct GPComplexCache
    cache_FloatVectors::Dict{Symbol, Vector{Float64}}
    cache_vectors::Dict{Symbol, Vector{ComplexF64}}
end
GPComplexCache() = GPComplexCache(Dict{Symbol,Vector{Float64}}(),Dict{Symbol,Vector{ComplexF64}}())

#------- Cache  ------- #

#------- Strategy ------- #

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

#----- Circulant Full convariance matrix --------#
# only use this strategy in case train and test input are identical
mutable struct CirculantFullCovarianceStrategy{N,R}
    n_inducing      :: N 
    rng             :: R 
    Afftmatrix 
    Ainvfftmatrix 
    cache           :: GPComplexCache 
end
CirculantFullCovarianceStrategy() = CirculantFullCovarianceStrategy(Int[],nothing,nothing,nothing,GPComplexCache()) 

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

# --------------- RFF ----------------- #
mutable struct RandomFourierFeature{N, R}
    n_inducing          :: N  
    rng                 :: R 
    n_samples           :: Int 
    ω                   :: AbstractArray 
    invKtrainNoise      :: AbstractArray 
    Z_xtrain            :: AbstractArray
    cache               :: GPCache
end
const RFF = RandomFourierFeature

RandomFourierFeature(n_samples) = RandomFourierFeature(Int[], nothing, n_samples, Float64[],Float64[1;;],Float64[1;;],GPCache()) 
#--------------- GP prediction ------------------#
function predictMVN end 

predictMVN(gpstrategy::CovarianceMatrixStrategy{<:FullCovarianceStrategy}, kernelfunc, meanfunc, xtrain, xtest, y, inducing)                 = fullcov(gpstrategy,kernelfunc, meanfunc, xtrain, xtest, y)
predictMVN(gpstrategy::CovarianceMatrixStrategy{<:DeterministicInducingConditional}, kernelfunc, meanfunc, xtrain, xtest, y, inducing)       = sor(gpstrategy,kernelfunc, meanfunc, xtrain, xtest, y, inducing)
predictMVN(gpstrategy::CovarianceMatrixStrategy{<:DeterministicTrainingConditional}, kernelfunc, meanfunc, xtrain, xtest, y, inducing)       = dtc(gpstrategy,kernelfunc, meanfunc, xtrain, xtest, y, inducing)
predictMVN(gpstrategy::CovarianceMatrixStrategy{<:FullyIndependentTrainingConditional}, kernelfunc, meanfunc, xtrain, xtest, y, inducing)    = fitc(gpstrategy,kernelfunc, meanfunc, xtrain, xtest, y, inducing)

predictMVN(gpstrategy::CovarianceMatrixStrategy{<:CirculantFullCovarianceStrategy}, kernelfunc, meanfunc, xtrain, xtest, y, inducing)        = cirfullcov(gpstrategy,kernelfunc, meanfunc, xtrain, xtest, y) #new strategy 
predictMVN(gpstrategy::CovarianceMatrixStrategy{<:RandomFourierFeature}, kernelfunc, meanfunc, xtrain, xtest, y, inducing)                   = rff(gpstrategy, kernelfunc, meanfunc, xtrain, xtest, y) # random fourier feature strategy 

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
    xtest==xtrain ? Σ = Matrix(Diagonal(Kff)) : Σ = Kff - mul_A_B_At!(cache, Kfy, invKtrain) 
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

## Circulant full covariance 
function cirfullcov(gpstrategy::CovarianceMatrixStrategy{<:CirculantFullCovarianceStrategy},kernelfunc, meanfunc, xtrain, xtest, y)
    N = length(xtest) #actual data length 
    N_cir = Int(2*(N-1))
    cache = gpstrategy.strategy.cache 

    circularVectorNoise = get!(cache.cache_FloatVectors, :CirVectorNoise, Vector{Float64}(undef, N_cir))
    Afftmatrix = gpstrategy.strategy.Afftmatrix
    Ainvfftmatrix = gpstrategy.strategy.Ainvfftmatrix
    afft = get!(cache.cache_vectors, :afft, Vector{ComplexF64}(undef, N))
    bfft = get!(cache.cache_vectors, :bfft, Vector{ComplexF64}(undef, N))
    cfft = get!(cache.cache_FloatVectors, :cfft,Vector{Float64}(undef, N_cir))
    seedVector_Ktest = get!(cache.cache_FloatVectors, :seedVector, Vector{Float64}(undef, N_cir)) #already circularized 

    #make Ktest circulant
    Circulant_Ktest = Circulant(seedVector_Ktest)

    mean_difference = y - meanfunc.(xtest)
    mean_xtest = meanfunc.(xtest)
    unit_vec = [1]
    append!(mean_difference,zeros(N-2))
    append!(mean_xtest,zeros(N-2))
    append!(unit_vec,zeros(2*N-3))
    #compute predictive mean and variance 
    μ_test = get!(cache.cache_FloatVectors, :predict_mean, Vector{Float64}(undef, N))
    for index = 1:N
        Vector_Ktest_addednoise = seedVector_Ktest + circularVectorNoise[index] * unit_vec
        v1 = fastinversemultiply(Vector_Ktest_addednoise,mean_difference,Afftmatrix, Ainvfftmatrix,afft,bfft,cfft)
        v1[N+1:end] = zeros(N-2)
        μ_test[index] = mean_xtest[index] + dot(Circulant_Ktest[:,index], v1)
    end
    return  μ_test, Matrix(Diagonal(Circulant_Ktest[1:N, 1:N]))
end


## Random fourier feature strategy 
function rff(gpstrategy::CovarianceMatrixStrategy{<:RandomFourierFeature}, kernelfunc, meanfunc, xtrain, xtest, y)
    cache = gpstrategy.strategy.cache
    ω = gpstrategy.strategy.ω
    InverseKtrainNoise = gpstrategy.strategy.invKtrainNoise
    Z_xtrain = gpstrategy.strategy.Z_xtrain
    xtrain == xtest ? Z_test = Z_xtrain : Z_test = featuremap(xtest, ω)

    Ktest = getcache(cache, (:Ktest,(length(xtest),length(xtrain))))
    Ktest_test = getcache(cache, (:Ktest, (length(xtest),length(xtest))))
    xtrain_transformed = getcache(cache, (:xtrain_transformed,length(xtrain)))
    μ_RFF = getcache(cache, (:μ_RFF,length(xtest)))
    result1 = getcache(cache, (:result1, size(InverseKtrainNoise,1)))
    result2 = getcache(cache, (:result2, size(Ktest,1)))

    mul!(Ktest,Z_test',Z_xtrain) 
    mul!(Ktest_test, Z_test', Z_test)
    map!(meanfunc, μ_RFF, xtest) 
    map!((x,z) -> x - meanfunc(z), xtrain_transformed, y, xtrain) 

    mul!(result1, InverseKtrainNoise, xtrain_transformed)
    mul!(result2, Ktest, result1) 

    map!(+, μ_RFF, μ_RFF, result2)

    xtrain == xtest ? Σ_RFF = Matrix{Float64}(Diagonal(Ktest_test)) : Σ_RFF = Ktest_test - mul_A_B_At!(cache,Ktest,InverseKtrainNoise) 
    return μ_RFF, Σ_RFF   
end
#------------------------------------- function for extracting the matrices for strategies  ------------------------------------# 
function extractmatrix! end 
function extractmatrix_change! end 

#--------------- full covariance strategy --------------------#
function extractmatrix!(gpstrategy::CovarianceMatrixStrategy{<:FullCovarianceStrategy}, kernel, x_train, Σ_noise, x_induc) 
    Kff = kernelmatrix(kernel,x_train,x_train)
    gpstrategy.strategy.invKff = cholinv(Kff + Diagonal(Σ_noise))

    return gpstrategy
end

function extractmatrix_change!(gpstrategy::CovarianceMatrixStrategy{<:FullCovarianceStrategy}, kernel, x_train, Σ_noise, x_induc) 
    # return extractmatrix!(gpstrategy,kernel, x_train, Σ_noise,x_induc)
    return gpstrategy
end

#-------------- RFF strategy --------------#
function extractmatrix!(gpstrategy::CovarianceMatrixStrategy{<:RandomFourierFeature}, kernel, x_train, Σ_noise, x_induc)
    cache = gpstrategy.strategy.cache
    n_samples = gpstrategy.strategy.n_samples
    p_ω = FourierTransformKernel(kernel)
    ω = rand(StableRNG(7), p_ω, n_samples)
    Z_xtrain = featuremap(x_train, ω)

    invΣnoise = inv(Σ_noise)
    A = getcache(cache,(:A, (size(invΣnoise,1),size(Z_xtrain,1))))
    B = getcache(cache,(:B, (size(Z_xtrain,1), size(Z_xtrain,1))))
    C = getcache(cache,(:C, (size(invΣnoise,1),size(Z_xtrain,1))))
    D = getcache(cache,(:D, (size(invΣnoise,1),size(invΣnoise,1))))
    InverseKtrainNoise = Woodbury_inversion(Z_xtrain,invΣnoise,A,B,C,D)
    gpstrategy.strategy.invKtrainNoise = InverseKtrainNoise
    gpstrategy.strategy.ω              = ω
    gpstrategy.strategy.Z_xtrain       = Z_xtrain 
    return gpstrategy
end

function extractmatrix_change!(gpstrategy::CovarianceMatrixStrategy{<:RandomFourierFeature},kernel, x_train, Σ_noise, x_induc)
    return gpstrategy
end
    # cache = gpstrategy.strategy.cache
    # ω = gpstrategy.strategy.ω
    # Z_xtrain = featuremap(x_train, ω)
    # invΣnoise = inv(Σ_noise)

    # A = getcache(cache,(:A, (size(invΣnoise,1),size(Z_xtrain,1))))
    # B = getcache(cache,(:B, (size(Z_xtrain,1), size(Z_xtrain,1))))
    # C = getcache(cache,(:C, (size(invΣnoise,1),size(Z_xtrain,1))))
    # D = getcache(cache,(:D, (size(invΣnoise,1),size(invΣnoise,1))))
    # InverseKtrainNoise = Woodbury_inversion(Z_xtrain,invΣnoise,A,B,C,D)
    # gpstrategy.strategy.invKtrainNoise = InverseKtrainNoise
    # gpstrategy.strategy.Z_xtrain = Z_xtrain
#--------------- Circulant Full Covariance Strategy -----------------#
function extractmatrix!(gpstrategy::CovarianceMatrixStrategy{<:CirculantFullCovarianceStrategy}, kernel, x_train, Σ_noise, x_induc)
    """
    kernel now is a stationary vector (not circularized yet)
    """
    #get the first column of the circulant covariance matrix. In this case, Ktrain = Ktest  
    cir_vector_K = circularize(kernel)
    cache = gpstrategy.strategy.cache

    #get fft structure 
    if isnothing(gpstrategy.strategy.Afftmatrix) 
        Afftmatrix,Ainvfftmatrix, afft, bfft, cfft  = returnFFTstructures(cir_vector_K)
        gpstrategy.strategy.Afftmatrix = Afftmatrix
        gpstrategy.strategy.Ainvfftmatrix = Ainvfftmatrix
        cache.cache_vectors = Dict(:afft => afft, :bfft => bfft)
        cache.cache_FloatVectors = Dict(:cfft => cfft)
    end
    #vector noise 
    diag_Σ_noise = diag(Σ_noise) 
    cir_diag_Σ_noise = circularize(diag_Σ_noise)

    cache.cache_FloatVectors[:CirVectorNoise] = cir_diag_Σ_noise 

    cache.cache_FloatVectors[:seedVector] = cir_vector_K
    return gpstrategy
end

function extractmatrix_change!(gpstrategy::CovarianceMatrixStrategy{<:CirculantFullCovarianceStrategy}, kernel, x_train, Σ_noise, x_induc)
    return gpstrategy 
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

# ------- helper functions for circulant covariance matrices --------- #
function circularize(arr)
    arrlength      = length(arr)
    circularlength = 2*(arrlength-1)
    
    circulararr    = Array{eltype(arr)}(undef,circularlength)
    
    @views circulararr[1:arrlength] .= view(arr,:,1)
    @inbounds [circulararr[i] = arr[arrlength - i%arrlength] for i=arrlength+1:circularlength]
    
    return circulararr
end


function returnFFTstructures(circularvector)
    N                   = length(circularvector)
    n                   = Int(N/2 + 1)
    Afftmatrix          = plan_rfft(circularvector)
    afft                = Vector{ComplexF64}(undef, n)
    bfft                = Vector{ComplexF64}(undef, n)
    cfft                = Vector{Float64}(undef, N)
    Ainvfftmatrix       = plan_irfft(afft, N)
    
    return Afftmatrix,Ainvfftmatrix,afft,bfft,cfft 
end


function fastmultiply(a, b , Afftmatrix, Ainvfftmatrix,afft,bfft,cfft)
    mul!(afft, Afftmatrix, a)
    mul!(bfft, Afftmatrix, b)
    
    afft .*= bfft
    return mul!(cfft, Ainvfftmatrix, afft)
end

function fastinversemultiply(a, b , Afftmatrix, Ainvfftmatrix,afft,bfft,cfft)
    mul!(afft, Afftmatrix, a)
    mul!(bfft, Afftmatrix, b)
    
    bfft ./= afft
    return mul!(cfft, Ainvfftmatrix, bfft)
end



function fastmahalanobis(a, b , Afftmatrix,afft,bfft)
    mul!(afft, Afftmatrix, a)
    mul!(bfft, Afftmatrix, b)
    return real((2*sum(afft .* abs.(bfft).^2) - (afft[1] * abs(bfft[1])^2 + afft[length(afft)] * abs(bfft[length(afft)])^2))/length(a))
end


function fastinvmahalanobis(a, b , Afftmatrix,afft,bfft)
    mul!(afft, Afftmatrix, a)
    mul!(bfft, Afftmatrix, b)
    
    return real((2*sum(abs.(bfft).^2 ./ afft) - (abs(bfft[1])^2/afft[1] + abs(bfft[length(afft)])^2/afft[length(afft)]))/length(a))
end

 """
 featuremap(X,ω)
 X here can be a vector (size N) of input 
 ω has size D (D is the number of samples)

 return matrix feature map Z_x size 2DxN
 """
 function featuremap(X, ω)
    D = length(ω)
    N = length(X)
    Z_x = Matrix{Float64}(undef, 2D, N)
    vector_z = Vector{Float64}(undef, 2D)
    for i=1:N
        vector_z[1:D] = [cos(ωi * X[i]) for ωi in ω]
        vector_z[D+1 : end] = [sin(ωi * X[i]) for ωi in ω] 
        Z_x[:,i] = vector_z
    end
    return Z_x / sqrt(D) 
end 

"""
Woodbury_inversion(Z_x::T,invΣ::T,A::T,B::T,C::T,D::T) where {T <: Matrix{Float64}}   
Returns an approximation to inv(K+Σ)
"""
function Woodbury_inversion(Z_x::T,invΣ::T,A::T,B::T,C::T,D::T) where {T <: Matrix{Float64}}
    mul!(A,invΣ,Z_x')
    mul!(B,Z_x,A)
    B = cholinv(B + I)
    mul!(C,A,B)
    mul!(D,C,A')
    map!(-,D,invΣ,D)
    return D
end

"""
This function takes FT of SE kernel and returns a distribution.
Here only 1-D case is taken into account.
"""
function FourierTransformKernel(::Type{<:SqExponentialKernel})
    return Normal(0., 1.)
end

function FourierTransformKernel(kernel::TransformedKernel{<:SqExponentialKernel, <:Transform})
    std = first(kernel.transform.s) 
    return Normal(0., std)
end

function FourierTransformKernel(kernel::ScaledKernel{<:TransformedKernel{<:SqExponentialKernel,<:Transform}})
    std = first(kernel.kernel.transform.s)
    return Normal(0., std)
end