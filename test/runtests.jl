module ReactiveMPTest

## https://discourse.julialang.org/t/generation-of-documentation-fails-qt-qpa-xcb-could-not-connect-to-display/60988
## https://gr-framework.org/workstations.html#no-output
ENV["GKSwstype"] = "100"

using Test, Documenter, ReactiveMP, Distributions
using TestSetExtensions
using Aqua

# Unregistered GraphPPL, do not commit this two lines, but use them to test ReactiveMP locally
# ENV["JULIA_PKG_USE_CLI_GIT"] = true
# import Pkg; Pkg.rm("GraphPPL"); Pkg.add(Pkg.PackageSpec(name="GraphPPL", rev="master"));

# DocMeta.setdocmeta!(ReactiveMP, :DocTestSetup, :(using ReactiveMP, Distributions); recursive=true)

# Example usage of a reduced testset
# julia --project --color=yes -e 'import Pkg; Pkg.test(test_args = [ "distributions:normal_mean_variance" ])'

function addtests(filename)
    key = filename_to_key(filename)
    if isempty(enabled_tests) || key in enabled_tests
        include(filename)
    end
end

function key_to_filename(key)
    splitted = split(key, ":")
    return if length(splitted) === 1
        string("test_", first(splitted), ".jl")
    else
        string(join(splitted[1:end-1], "/"), "/test_", splitted[end], ".jl")
    end
end

function filename_to_key(filename)
    splitted = split(filename, "/")
    if length(splitted) === 1
        return replace(replace(first(splitted), ".jl" => ""), "test_" => "")
    else
        path, name = splitted[1:end-1], splitted[end]
        return string(join(path, ":"), ":", replace(replace(name, ".jl" => ""), "test_" => ""))
    end
end

enabled_tests = lowercase.(ARGS)

if isempty(enabled_tests)
    println("Running all tests...")
    # `project_toml_formatting` is broken on CI, revise at some point
    Aqua.test_all(ReactiveMP; ambiguities = false, project_toml_formatting = false)
    # doctest(ReactiveMP)
else
    println("Running specific tests: $enabled_tests")
end

@testset ExtendedTestSet "ReactiveMP" begin
    function key_to_filename(key)
        splitted = split(key, ":")
        return if length(splitted) === 1
            string("test_", first(splitted), ".jl")
        else
            string(join(splitted[1:end-1], "/"), "/test_", splitted[end], ".jl")
        end
    end

    function filename_to_key(filename)
        splitted = split(filename, "/")
        if length(splitted) === 1
            return replace(replace(first(splitted), ".jl" => ""), "test_" => "")
        else
            path, name = splitted[1:end-1], splitted[end]
            return string(join(path, ":"), ":", replace(replace(name, ".jl" => ""), "test_" => ""))
        end
    end

    function addtests(filename)
        key = filename_to_key(filename)
        if isempty(enabled_tests) || key in enabled_tests
            include(filename)
        end
    end

    @testset "Testset helpers" begin
        @test key_to_filename(filename_to_key("distributions/test_normal_mean_variance.jl")) ==
              "distributions/test_normal_mean_variance.jl"
        @test filename_to_key(key_to_filename("distributions:normal_mean_variance")) ==
              "distributions:normal_mean_variance"
        @test key_to_filename(filename_to_key("test_message.jl")) == "test_message.jl"
        @test filename_to_key(key_to_filename("message")) == "message"
    end

    addtests("algebra/test_correction.jl")
    addtests("algebra/test_helpers.jl")
    addtests("algebra/test_permutation_matrix.jl")
    addtests("algebra/test_standard_basis_vector.jl")

    addtests("test_model.jl")
    addtests("test_math.jl")
    addtests("test_helpers.jl")
    addtests("test_score.jl")

    addtests("constraints/spec/test_factorisation_spec.jl")
    addtests("constraints/spec/test_form_spec.jl")
    addtests("constraints/form/test_form_point_mass.jl")
    addtests("constraints/prod/test_prod_final.jl")
    addtests("constraints/prod/test_prod_generic.jl")
    addtests("constraints/meta/test_meta.jl")

    addtests("test_distributions.jl")
    addtests("distributions/test_common.jl")
    addtests("distributions/test_bernoulli.jl")
    addtests("distributions/test_beta.jl")
    addtests("distributions/test_categorical.jl")
    addtests("distributions/test_contingency.jl")
    addtests("distributions/test_exp_linear_quadratic.jl")
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
    addtests("distributions/test_normal_natural_parameters.jl")
    addtests("distributions/test_pointmass.jl")
    addtests("distributions/test_wishart.jl")
    addtests("distributions/test_wishart_inverse.jl")
    addtests("distributions/test_sample_list.jl")

    addtests("test_message.jl")

    addtests("test_variable.jl")
    addtests("variables/test_constant.jl")
    addtests("variables/test_data.jl")
    addtests("variables/test_random.jl")

    addtests("test_node.jl")
    addtests("nodes/flow/test_flow.jl")
    addtests("nodes/test_addition.jl")
    addtests("nodes/test_bifm.jl")
    addtests("nodes/test_bifm_helper.jl")
    addtests("nodes/test_subtraction.jl")
    addtests("nodes/test_probit.jl")
    addtests("nodes/test_autoregressive.jl")
    addtests("nodes/test_normal_mean_precision.jl")
    addtests("nodes/test_normal_mean_variance.jl")
    addtests("nodes/test_mv_normal_mean_precision.jl")
    addtests("nodes/test_mv_normal_mean_covariance.jl")
    addtests("nodes/test_poisson.jl")
    addtests("nodes/test_wishart_inverse.jl")
    addtests("nodes/test_or.jl")
    addtests("nodes/test_not.jl")
    addtests("nodes/test_and.jl")
    addtests("nodes/test_implication.jl")

    addtests("rules/uniform/test_out.jl")

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

    addtests("rules/normal_mixture/test_out.jl")

    addtests("rules/subtraction/test_marginals.jl")
    addtests("rules/subtraction/test_in1.jl")
    addtests("rules/subtraction/test_in2.jl")
    addtests("rules/subtraction/test_out.jl")

    addtests("rules/bernoulli/test_out.jl")
    addtests("rules/bernoulli/test_p.jl")
    addtests("rules/bernoulli/test_marginals.jl")

    addtests("rules/beta/test_out.jl")
    addtests("rules/beta/test_marginals.jl")

    addtests("rules/dot_product/test_out.jl")
    addtests("rules/dot_product/test_in1.jl")
    addtests("rules/dot_product/test_in2.jl")
    addtests("rules/dot_product/test_marginals.jl")

    addtests("rules/normal_mean_variance/test_out.jl")
    addtests("rules/normal_mean_variance/test_mean.jl")
    addtests("rules/normal_mean_variance/test_var.jl")

    addtests("rules/normal_mean_precision/test_out.jl")
    addtests("rules/normal_mean_precision/test_mean.jl")
    addtests("rules/normal_mean_precision/test_precision.jl")

    addtests("rules/mv_normal_mean_covariance/test_out.jl")
    addtests("rules/mv_normal_mean_covariance/test_mean.jl")
    addtests("rules/mv_normal_mean_covariance/test_covariance.jl")

    addtests("rules/mv_normal_mean_precision/test_out.jl")
    addtests("rules/mv_normal_mean_precision/test_mean.jl")
    addtests("rules/mv_normal_mean_precision/test_precision.jl")

    addtests("rules/probit/test_out.jl")
    addtests("rules/probit/test_in.jl")

    addtests("rules/wishart/test_marginals.jl")
    addtests("rules/wishart/test_out.jl")

    addtests("rules/wishart_inverse/test_marginals.jl")
    addtests("rules/wishart_inverse/test_out.jl")

    addtests("rules/poisson/test_l.jl")
    addtests("rules/poisson/test_marginals.jl")
    addtests("rules/poisson/test_out.jl")

    addtests("rules/or/test_out.jl")
    addtests("rules/or/test_in1.jl")
    addtests("rules/or/test_in2.jl")
    addtests("rules/or/test_marginals.jl")

    addtests("rules/not/test_out.jl")
    addtests("rules/not/test_in.jl")
    addtests("rules/not/test_marginals.jl")

    addtests("rules/and/test_out.jl")
    addtests("rules/and/test_in1.jl")
    addtests("rules/and/test_in2.jl")
    addtests("rules/and/test_marginals.jl")

    addtests("rules/implication/test_out.jl")
    addtests("rules/implication/test_in1.jl")
    addtests("rules/implication/test_in2.jl")
    addtests("rules/implication/test_marginals.jl")

    addtests("models/test_lgssm.jl")
    addtests("models/test_hgf.jl")
    addtests("models/test_ar.jl")
    addtests("models/test_gmm.jl")
    addtests("models/test_hmm.jl")
    addtests("models/test_linreg.jl")
    addtests("models/test_mv_iid.jl")
    addtests("models/test_probit.jl")
    addtests("models/test_aliases.jl")
    addtests("models/test_cvi.jl")
end

end
