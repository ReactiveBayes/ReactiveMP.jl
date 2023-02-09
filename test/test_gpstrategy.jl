module ReactiveMPGPstrategyTest 

using Test
using ReactiveMP
using KernelFunctions, Random 
using LinearAlgebra
import ReactiveMP: GPCache
import ReactiveMP: kernelmatrix!, getcache, mul_A_B_At!, extractmatrix!, extractmatrix_change!
import ReactiveMP: CovarianceMatrixStrategy, FullCovarianceStrategy, DeterministicInducingConditional, DIC, SoR, SubsetOfRegressors
import ReactiveMP: DeterministicTrainingConditional, DTC, FullyIndependentTrainingConditional, FITC
import ReactiveMP: predictMVN, fullcov, dtc, fitc, sor

@testset "GPstrategy" begin 
    @testset "GPcache" begin
        x1 = [1.5, 2.4]
        x2 = [1., 5.3]

        gpcache_test = GPCache()
        @test typeof(gpcache_test) == GPCache

        gpcache = GPCache(Dict((:matrix,(2,2)) => [1. 0.; 0. 2.]), Dict((:vector,(2)) => [3.,4.]))
        
        @test getcache(gpcache, (:matrix,(2,2))) == [1. 0.;0. 2.]
        @test getcache(gpcache, (:vector, (2))) == [3., 4.]

        changed_matrix_cache = kernelmatrix!(gpcache, :matrix, SqExponentialKernel(),x1,x2)
        @test getcache(gpcache, (:matrix,(2,2))) == changed_matrix_cache
        @test getcache(gpcache, (:matrix,(2,2))) == kernelmatrix(SqExponentialKernel(),x1,x2)
        @test changed_matrix_cache == kernelmatrix(SqExponentialKernel(),x1,x2)
        A_vec = [1 3]
        B_mat = [2 0;0 3]
        matrix_ABAt = mul_A_B_At!(gpcache,A_vec,B_mat)
        @test matrix_ABAt ==  A_vec * B_mat * A_vec'
    end

    @testset "GPprediction" begin
        n = 100
        N = 50
        l = 0.6  #lengthscale
        n_inducing = 20
        gt_function = (x) -> sinc(x)
        xtest = collect(range(-5., 5.; length=n))
        y_evaluated_at_xtest = gt_function.(xtest)
        pos = sort(randperm(n)[1:N]); # position where we observe data
        xtrain = xtest[pos]
        ytrain = y_evaluated_at_xtest[pos] + 0.5*rand(N) #our observation 
        inducing_pos = sort(randperm(N)[1:n_inducing])
        inducing_points = xtrain[inducing_pos]
        gp_kernel_func = with_lengthscale(SqExponentialKernel(),l)
        gp_meanfunc = (x) -> 0.0
        strategy_fullcov = CovarianceMatrixStrategy(FullCovarianceStrategy())
        strategy_sor = CovarianceMatrixStrategy(SoR(n_inducing))
        strategy_dtc = CovarianceMatrixStrategy(DTC(n_inducing))
        strategy_fitc = CovarianceMatrixStrategy(FITC(n_inducing))

        @testset "FullcovariancePredictMVN" begin
            K_test_train = kernelmatrix(gp_kernel_func, xtest, xtrain)
            K_test = kernelmatrix(gp_kernel_func,xtest,xtest)
            Ktrain = kernelmatrix(gp_kernel_func,xtrain,xtrain)
            invK_train = cholinv(Ktrain + 1e-6*diageye(length(xtrain)))
            μ_fullcov_gt = gp_meanfunc.(xtest) + K_test_train * invK_train * (ytrain - gp_meanfunc.(xtrain))
            Σ_fullcov_gt = K_test - K_test_train * invK_train * K_test_train'

            extractmatrix!(strategy_fullcov, gp_kernel_func, xtrain, 1e-6*diageye(length(xtrain)), inducing_points) #this function goes along with the predictMVN
            μ_fullcov, Σ_fullcov = predictMVN(strategy_fullcov, gp_kernel_func,gp_meanfunc,xtrain,xtest,ytrain,inducing_points) 

            @test μ_fullcov == μ_fullcov_gt
            @test Σ_fullcov == Σ_fullcov_gt

            #now use test to predict train 
            xtrain_sample = xtrain[3]
            k_train_sample = kernelmatrix(gp_kernel_func,[xtrain_sample],[xtrain_sample])
            K_sample_test = kernelmatrix(gp_kernel_func, [xtrain_sample],xtest)
            invK_test = cholinv(K_test + 1e-6*diageye(length(xtest)))
            μ_train_fullcov_gt = gp_meanfunc(xtrain_sample) .+ K_sample_test * invK_test * (y_evaluated_at_xtest - gp_meanfunc.(xtest))
            Σ_train_fullcov_gt = k_train_sample - K_sample_test * invK_test * K_sample_test'

            extractmatrix_change!(strategy_fullcov, gp_kernel_func, xtest, 1e-6*diageye(length(xtest)), inducing_points)
            μ_train_fullcov, Σ_train_fullcov = predictMVN(strategy_fullcov, gp_kernel_func,gp_meanfunc,xtest,[xtrain_sample],y_evaluated_at_xtest,inducing_points) 
            @test Σ_train_fullcov == Σ_train_fullcov_gt
            @test μ_train_fullcov == μ_train_fullcov_gt 
        end

        @testset "SoRPredictMVN" begin
            Kuu = kernelmatrix(gp_kernel_func,inducing_points,inducing_points)
            invKuu = cholinv(Kuu)
            Ktest_inducing = kernelmatrix(gp_kernel_func,xtest,inducing_points)
            Ktrain_inducing = kernelmatrix(gp_kernel_func,xtrain,inducing_points)

            Q_test_train = Ktest_inducing * invKuu * Ktrain_inducing'
            Q_test = Ktest_inducing * invKuu * Ktest_inducing'
            Q_train = Ktrain_inducing * invKuu * Ktrain_inducing'

            μ_sor_gt = gp_meanfunc.(xtest) + Q_test_train * cholinv(Q_train + 1e-3*diageye(length(xtrain))) * (ytrain - gp_meanfunc.(xtrain))
            Σ_sor_gt = Q_test - Q_test_train * cholinv(Q_train + 1e-3*diageye(length(xtrain))) * Q_test_train'

            extractmatrix!(strategy_sor, gp_kernel_func, xtrain, 1e-3*diageye(length(xtrain)), inducing_points) #this function goes along with the predictMVN
            μ_sor, Σ_sor = predictMVN(strategy_sor, gp_kernel_func,gp_meanfunc,xtrain,xtest,ytrain,inducing_points) 

            @test norm(μ_sor - μ_sor_gt,2) < 1e-5
            @test norm(Σ_sor - Σ_sor_gt) < 1e-5

            #now use test to predict train 
            # xtrain_sample = xtrain[3]
            # k_train_sample = kernelmatrix(gp_kernel_func,[xtrain_sample],[xtrain_sample])
            # K_sample_test = kernelmatrix(gp_kernel_func, [xtrain_sample],xtest)
            # invK_test = cholinv(K_test + 1e-6*diageye(length(xtest)))
            # μ_train_sor_gt = gp_meanfunc(xtrain_sample) .+ K_sample_test * invK_test * (y_evaluated_at_xtest - gp_meanfunc.(xtest))
            # Σ_train_sor_gt = k_train_sample - K_sample_test * invK_test * K_sample_test'

            # extractmatrix_change!(strategy_sor, gp_kernel_func, xtest, 1e-6*diageye(length(xtest)), inducing_points)
            # μ_train_sor, Σ_train_sor = predictMVN(strategy_sor, gp_kernel_func,gp_meanfunc,xtest,[xtrain_sample],y_evaluated_at_xtest,inducing_points) 
            # @test Σ_train_sor == Σ_train_sor_gt
            # @test μ_train_sor == μ_train_sor_gt 
        end
    end
end

end