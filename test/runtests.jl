module ReactiveMPTest

using Test, Documenter, ReactiveMP
using TestSetExtensions

using Aqua
Aqua.test_all(ReactiveMP; ambiguities=false)

include("test_helpers.jl")

using .ReactiveMPTestingHelpers

# doctest(ReactiveMP)

@testset ExtendedTestSet "ReactiveMP" begin

    @testset "Testset helpers" begin
        @test key_to_filename(filename_to_key("distributions/test_normal_mean_variance.jl")) == "distributions/test_normal_mean_variance.jl"
        @test filename_to_key(key_to_filename("distributions:normal_mean_variance")) == "distributions:normal_mean_variance"
        @test key_to_filename(filename_to_key("test_message.jl")) == "test_message.jl"
        @test filename_to_key(key_to_filename("message")) == "message"
    end

    addtests("algebra/test_correction.jl")
    addtests("algebra/test_flatten_tuple.jl")
    addtests("algebra/test_permutation_matrix.jl")

    addtests("test_math.jl")

    addtests("constraints/prod/test_prod_final.jl")

    addtests("test_distributions.jl")
    addtests("distributions/test_common.jl")
    addtests("distributions/test_bernoulli.jl")
    addtests("distributions/test_beta.jl")
    addtests("distributions/test_categorical.jl")
    addtests("distributions/test_contingency.jl")
    addtests("distributions/test_dirichlet_matrix.jl")
    addtests("distributions/test_dirichlet.jl")
    addtests("distributions/test_gamma.jl")
    addtests("distributions/test_mv_normal_mean_covariance.jl")
    addtests("distributions/test_mv_normal_mean_precision.jl")
    addtests("distributions/test_mv_normal_weighted_mean_precision.jl")
    addtests("distributions/test_normal_mean_variance.jl")
    addtests("distributions/test_normal_mean_precision.jl")
    addtests("distributions/test_normal_weighted_mean_precision.jl")
    addtests("distributions/test_normal.jl")
    addtests("distributions/test_pointmass.jl")
    addtests("distributions/test_wishart.jl")
    addtests("distributions/test_sample_list.jl")

    addtests("test_message.jl")
    
    addtests("test_variable.jl")

    addtests("test_node.jl")
    addtests("nodes/flow/test_flow.jl")
    addtests("nodes/test_addition.jl")
    addtests("nodes/test_bifm.jl")
    addtests("nodes/test_bifm_helper.jl")
    addtests("nodes/test_subtraction.jl")
    addtests("nodes/test_probit.jl")
    
    addtests("rules/flow/test_marginals.jl")
    addtests("rules/flow/test_in.jl")
    addtests("rules/flow/test_out.jl")

    addtests("rules/addition/test_marginals.jl")
    addtests("rules/addition/test_in1.jl")
    addtests("rules/addition/test_in2.jl")
    addtests("rules/addition/test_out.jl")

    addtests("rules/bifm/test_marginals.jl")
    addtests("rules/bifm/test_in.jl")
    addtests("rules/bifm/test_out.jl")
    addtests("rules/bifm/test_zprev.jl")
    addtests("rules/bifm/test_znext.jl")

    addtests("rules/bifm_helper/test_in.jl")
    addtests("rules/bifm_helper/test_out.jl")

		
    addtests("rules/subtraction/test_marginals.jl")
    addtests("rules/subtraction/test_in1.jl")
    addtests("rules/subtraction/test_in2.jl")
    addtests("rules/subtraction/test_out.jl")

    addtests("rules/bernoulli/test_out.jl")
    addtests("rules/bernoulli/test_p.jl")
    addtests("rules/bernoulli/test_marginals.jl")
    
    addtests("rules/beta/test_out.jl")
    addtests("rules/beta/test_marginals.jl")

    addtests("rules/normal_mean_variance/test_out.jl")
    addtests("rules/normal_mean_variance/test_mean.jl")
    addtests("rules/normal_mean_variance/test_var.jl")

    addtests("rules/normal_mean_precision/test_out.jl")
    addtests("rules/normal_mean_precision/test_mean.jl")
    addtests("rules/normal_mean_precision/test_precision.jl")

    addtests("rules/mv_normal_mean_covariance/test_out.jl")
    addtests("rules/mv_normal_mean_covariance/test_mean.jl")

    addtests("rules/mv_normal_mean_precision/test_out.jl")
    addtests("rules/mv_normal_mean_precision/test_mean.jl")
    addtests("rules/mv_normal_mean_precision/test_precision.jl")  

    addtests("rules/probit/test_out.jl")
    addtests("rules/probit/test_in.jl")

    addtests("rules/wishart/test_marginals.jl")
    addtests("rules/wishart/test_out.jl")

end

end
