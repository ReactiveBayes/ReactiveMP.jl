

## https://discourse.julialang.org/t/generation-of-documentation-fails-qt-qpa-xcb-could-not-connect-to-display/60988
## https://gr-framework.org/workstations.html#no-output
ENV["GKSwstype"] = "100"

using Distributed

const WORKER_END_TOKEN = "[¬]" # Supposed to be somewhat unique
const worker_io_lock   = ReentrantLock()
const worker_ios       = Dict()

worker_io(ident) = get!(() -> IOBuffer(), worker_ios, ident)

# Dynamically overwrite default worker's `print` function for better control over stdout
Distributed.redirect_worker_output(ident, stream) = begin
    task = @async while !eof(stream)
        lock(worker_io_lock) do 
            line = readline(stream)
            io   = worker_io(ident)

            if startswith(line, WORKER_END_TOKEN)
                println(stdout, String(take!(io)))
                flush(stdout)
            else
                write(io, line, "\n")
            end
        end
    end
    Base.errormonitor(task)
end

# Unregistered GraphPPL, do not commit this two lines, but use them to test ReactiveMP locally
# ENV["JULIA_PKG_USE_CLI_GIT"] = true
# import Pkg; Pkg.rm("GraphPPL"); Pkg.add(Pkg.PackageSpec(name="GraphPPL", rev="master"));

# DocMeta.setdocmeta!(ReactiveMP, :DocTestSetup, :(using ReactiveMP, Distributions); recursive=true)

# Example usage of a reduced testset
# julia --project --color=yes -e 'import Pkg; Pkg.test(test_args = [ "distributions:normal_mean_variance" ])'

addprocs(4)

@everywhere using Test, Documenter, ReactiveMP, Distributions
@everywhere using TestSetExtensions, Suppressor
@everywhere using Aqua

import Base: wait

enabled_tests = lowercase.(ARGS)

mutable struct TestRunner
    enabled_tests
    test_tasks 
    workerpool

    function TestRunner(ARGS)
        enabled_tests = lowercase.(ARGS)
        test_tasks    = []
        return new(enabled_tests, test_tasks, WorkerPool(collect(2:nprocs())))
    end
end

function Base.wait(testrunner::TestRunner)
    exceptions = []

    foreach(testrunner.test_tasks) do task 
        try 
            (filename, ) = fetch(task)
        catch exception
            push!(exceptions, exception)
        end
    end

    if !isempty(exceptions)
        println("Tests have failed: ")
        foreach(exceptions) do exception 
            println("="^80, "\n", exception)
        end
        error("Tests have failed")
    end
end

testrunner = TestRunner(ARGS)

println("`TestRunner` has been created. The number of available procs is $(nprocs()).")

@everywhere tasklock = ReentrantLock()

function addtests(testrunner::TestRunner, filename)
    key = filename_to_key(filename)
    if isempty(enabled_tests) || key in enabled_tests
        task = remotecall(testrunner.workerpool, filename) do filename 
            lock(tasklock) do
                include(filename)
                println(WORKER_END_TOKEN)
            end
            return (filename, )
        end
        push!(testrunner.test_tasks, task)
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

if isempty(enabled_tests)
    println("Running all tests...")
    # `project_toml_formatting` is broken on CI, revise at some point
    Aqua.test_all(ReactiveMP; ambiguities = false, project_toml_formatting = false)
    # doctest(ReactiveMP)
else
    println("Running specific tests: $enabled_tests")
end

@testset ExtendedTestSet "ReactiveMP" begin

    @testset "Testset helpers" begin
        @test key_to_filename(filename_to_key("distributions/test_normal_mean_variance.jl")) ==
              "distributions/test_normal_mean_variance.jl"
        @test filename_to_key(key_to_filename("distributions:normal_mean_variance")) ==
              "distributions:normal_mean_variance"
        @test key_to_filename(filename_to_key("test_message.jl")) == "test_message.jl"
        @test filename_to_key(key_to_filename("message")) == "message"
    end

    println("") # New line for the `testrunner`

    addtests(testrunner, "algebra/test_correction.jl")
    addtests(testrunner, "algebra/test_helpers.jl")
    addtests(testrunner, "algebra/test_permutation_matrix.jl")
    addtests(testrunner, "algebra/test_standard_basis_vector.jl")

    addtests(testrunner, "test_model.jl")
    addtests(testrunner, "test_math.jl")
    addtests(testrunner, "test_helpers.jl")
    addtests(testrunner, "test_score.jl")

    addtests(testrunner, "constraints/spec/test_factorisation_spec.jl")
    addtests(testrunner, "constraints/spec/test_form_spec.jl")
    addtests(testrunner, "constraints/form/test_form_point_mass.jl")
    addtests(testrunner, "constraints/prod/test_prod_final.jl")
    addtests(testrunner, "constraints/prod/test_prod_generic.jl")
    addtests(testrunner, "constraints/meta/test_meta.jl")

    addtests(testrunner, "test_distributions.jl")
    addtests(testrunner, "distributions/test_common.jl")
    addtests(testrunner, "distributions/test_bernoulli.jl")
    addtests(testrunner, "distributions/test_beta.jl")
    addtests(testrunner, "distributions/test_categorical.jl")
    addtests(testrunner, "distributions/test_contingency.jl")
    addtests(testrunner, "distributions/test_exp_linear_quadratic.jl")
    addtests(testrunner, "distributions/test_dirichlet_matrix.jl")
    addtests(testrunner, "distributions/test_dirichlet.jl")
    addtests(testrunner, "distributions/test_gamma.jl")
    addtests(testrunner, "distributions/test_mv_normal_mean_covariance.jl")
    addtests(testrunner, "distributions/test_mv_normal_mean_precision.jl")
    addtests(testrunner, "distributions/test_mv_normal_weighted_mean_precision.jl")
    addtests(testrunner, "distributions/test_normal_mean_variance.jl")
    addtests(testrunner, "distributions/test_normal_mean_precision.jl")
    addtests(testrunner, "distributions/test_normal_weighted_mean_precision.jl")
    addtests(testrunner, "distributions/test_normal.jl")
    addtests(testrunner, "distributions/test_pointmass.jl")
    addtests(testrunner, "distributions/test_wishart.jl")
    addtests(testrunner, "distributions/test_wishart_inverse.jl")
    addtests(testrunner, "distributions/test_sample_list.jl")

    addtests(testrunner, "test_message.jl")

    addtests(testrunner, "test_variable.jl")
    addtests(testrunner, "variables/test_constant.jl")
    addtests(testrunner, "variables/test_data.jl")
    addtests(testrunner, "variables/test_random.jl")

    addtests(testrunner, "test_node.jl")
    addtests(testrunner, "nodes/flow/test_flow.jl")
    addtests(testrunner, "nodes/test_addition.jl")
    addtests(testrunner, "nodes/test_bifm.jl")
    addtests(testrunner, "nodes/test_bifm_helper.jl")
    addtests(testrunner, "nodes/test_subtraction.jl")
    addtests(testrunner, "nodes/test_probit.jl")
    addtests(testrunner, "nodes/test_autoregressive.jl")
    addtests(testrunner, "nodes/test_normal_mean_precision.jl")
    addtests(testrunner, "nodes/test_normal_mean_variance.jl")
    addtests(testrunner, "nodes/test_mv_normal_mean_precision.jl")
    addtests(testrunner, "nodes/test_mv_normal_mean_covariance.jl")
    addtests(testrunner, "nodes/test_poisson.jl")
    addtests(testrunner, "nodes/test_wishart_inverse.jl")
    addtests(testrunner, "nodes/test_or.jl")
    addtests(testrunner, "nodes/test_not.jl")
    addtests(testrunner, "nodes/test_and.jl")
    addtests(testrunner, "nodes/test_implication.jl")

    addtests(testrunner, "rules/uniform/test_out.jl")

    addtests(testrunner, "rules/flow/test_marginals.jl")
    addtests(testrunner, "rules/flow/test_in.jl")
    addtests(testrunner, "rules/flow/test_out.jl")

    addtests(testrunner, "rules/addition/test_marginals.jl")
    addtests(testrunner, "rules/addition/test_in1.jl")
    addtests(testrunner, "rules/addition/test_in2.jl")
    addtests(testrunner, "rules/addition/test_out.jl")

    addtests(testrunner, "rules/bifm/test_marginals.jl")
    addtests(testrunner, "rules/bifm/test_in.jl")
    addtests(testrunner, "rules/bifm/test_out.jl")
    addtests(testrunner, "rules/bifm/test_zprev.jl")
    addtests(testrunner, "rules/bifm/test_znext.jl")

    addtests(testrunner, "rules/bifm_helper/test_in.jl")
    addtests(testrunner, "rules/bifm_helper/test_out.jl")

    addtests(testrunner, "rules/normal_mixture/test_out.jl")

    addtests(testrunner, "rules/subtraction/test_marginals.jl")
    addtests(testrunner, "rules/subtraction/test_in1.jl")
    addtests(testrunner, "rules/subtraction/test_in2.jl")
    addtests(testrunner, "rules/subtraction/test_out.jl")

    addtests(testrunner, "rules/bernoulli/test_out.jl")
    addtests(testrunner, "rules/bernoulli/test_p.jl")
    addtests(testrunner, "rules/bernoulli/test_marginals.jl")

    addtests(testrunner, "rules/beta/test_out.jl")
    addtests(testrunner, "rules/beta/test_marginals.jl")

    addtests(testrunner, "rules/dot_product/test_out.jl")
    addtests(testrunner, "rules/dot_product/test_in1.jl")
    addtests(testrunner, "rules/dot_product/test_in2.jl")
    addtests(testrunner, "rules/dot_product/test_marginals.jl")

    addtests(testrunner, "rules/normal_mean_variance/test_out.jl")
    addtests(testrunner, "rules/normal_mean_variance/test_mean.jl")
    addtests(testrunner, "rules/normal_mean_variance/test_var.jl")

    addtests(testrunner, "rules/normal_mean_precision/test_out.jl")
    addtests(testrunner, "rules/normal_mean_precision/test_mean.jl")
    addtests(testrunner, "rules/normal_mean_precision/test_precision.jl")

    addtests(testrunner, "rules/mv_normal_mean_covariance/test_out.jl")
    addtests(testrunner, "rules/mv_normal_mean_covariance/test_mean.jl")
    addtests(testrunner, "rules/mv_normal_mean_covariance/test_covariance.jl")

    addtests(testrunner, "rules/mv_normal_mean_precision/test_out.jl")
    addtests(testrunner, "rules/mv_normal_mean_precision/test_mean.jl")
    addtests(testrunner, "rules/mv_normal_mean_precision/test_precision.jl")

    addtests(testrunner, "rules/probit/test_out.jl")
    addtests(testrunner, "rules/probit/test_in.jl")

    addtests(testrunner, "rules/wishart/test_marginals.jl")
    addtests(testrunner, "rules/wishart/test_out.jl")

    addtests(testrunner, "rules/wishart_inverse/test_marginals.jl")
    addtests(testrunner, "rules/wishart_inverse/test_out.jl")

    addtests(testrunner, "rules/poisson/test_l.jl")
    addtests(testrunner, "rules/poisson/test_marginals.jl")
    addtests(testrunner, "rules/poisson/test_out.jl")

    addtests(testrunner, "rules/or/test_out.jl")
    addtests(testrunner, "rules/or/test_in1.jl")
    addtests(testrunner, "rules/or/test_in2.jl")
    addtests(testrunner, "rules/or/test_marginals.jl")

    addtests(testrunner, "rules/not/test_out.jl")
    addtests(testrunner, "rules/not/test_in.jl")
    addtests(testrunner, "rules/not/test_marginals.jl")

    addtests(testrunner, "rules/and/test_out.jl")
    addtests(testrunner, "rules/and/test_in1.jl")
    addtests(testrunner, "rules/and/test_in2.jl")
    addtests(testrunner, "rules/and/test_marginals.jl")

    addtests(testrunner, "rules/implication/test_out.jl")
    addtests(testrunner, "rules/implication/test_in1.jl")
    addtests(testrunner, "rules/implication/test_in2.jl")
    addtests(testrunner, "rules/implication/test_marginals.jl")

    addtests(testrunner, "models/test_lgssm.jl")
    addtests(testrunner, "models/test_hgf.jl")
    addtests(testrunner, "models/test_ar.jl")
    addtests(testrunner, "models/test_gmm.jl")
    addtests(testrunner, "models/test_hmm.jl")
    addtests(testrunner, "models/test_linreg.jl")
    addtests(testrunner, "models/test_mv_iid.jl")
    addtests(testrunner, "models/test_probit.jl")

    wait(testrunner)

end