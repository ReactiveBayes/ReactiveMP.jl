
## https://discourse.julialang.org/t/generation-of-documentation-fails-qt-qpa-xcb-could-not-connect-to-display/60988
## https://gr-framework.org/workstations.html#no-output
ENV["GKSwstype"] = "100"

using Distributed

const worker_io_lock = ReentrantLock()
const worker_ios     = Dict()

worker_io(ident) = get!(() -> IOBuffer(), worker_ios, string(ident))

# Dynamically overwrite default worker's `print` function for better control over stdout
Distributed.redirect_worker_output(ident, stream) = begin
    task = @async while !eof(stream)
        line = readline(stream)
        lock(worker_io_lock) do
            io = worker_io(ident)
            write(io, line, "\n")
        end
    end
    @static if VERSION >= v"1.7"
        Base.errormonitor(task)
    end
end

# This function prints `worker's` standard output into the global standard output
function flush_workerio(ident)
    lock(worker_io_lock) do
        wio = worker_io(ident)
        str = String(take!(wio))
        println(stdout, str)
        flush(stdout)
    end
end

# Unregistered GraphPPL, do not commit this two lines, but use them to test ReactiveMP locally
# ENV["JULIA_PKG_USE_CLI_GIT"] = true
# import Pkg; Pkg.rm("GraphPPL"); Pkg.add(Pkg.PackageSpec(name="GraphPPL", rev="master"));

# DocMeta.setdocmeta!(ReactiveMP, :DocTestSetup, :(using ReactiveMP, Distributions); recursive=true)

# Example usage of a reduced testset
# julia --project --color=yes -e 'import Pkg; Pkg.test(test_args = [ "distributions:normal_mean_variance" ])'

# Makes it hard to use your computer if Julia occupies all cpus, so we max at 4
# GitHub actions has 2 cores in most of the cases 
addprocs(min(Sys.CPU_THREADS, 4))

@everywhere using Test, Documenter, ReactiveMP, Distributions
@everywhere using TestSetExtensions

import Base: wait

mutable struct TestRunner
    enabled_tests
    found_tests
    test_tasks
    workerpool
    jobschannel
    exschannel
    iochannel

    function TestRunner(ARGS)
        enabled_tests = lowercase.(ARGS)
        found_tests = Dict(map(test -> test => false, enabled_tests))
        test_tasks = []
        jobschannel = RemoteChannel(() -> Channel(Inf), myid()) # Channel for jobs
        exschannel = RemoteChannel(() -> Channel(Inf), myid()) # Channel for exceptions
        iochannel = RemoteChannel(() -> Channel(0), myid())
        @async begin
            while isopen(iochannel)
                ident = take!(iochannel)
                flush_workerio(ident)
            end
        end
        return new(enabled_tests, found_tests, test_tasks, 2:nprocs(), jobschannel, exschannel, iochannel)
    end
end

function Base.run(testrunner::TestRunner)
    println("") # New line for 'better' alignment of the `testrunner` results

    foreach(testrunner.workerpool) do worker
        # For each worker we create a `nothing` token in the `jobschannel`
        # This token indicates that there are no other jobs left
        put!(testrunner.jobschannel, nothing)
        # We create a remote call for another Julia process to execute our test with `include(filename)`
        task = remotecall(worker, testrunner.jobschannel, testrunner.exschannel, testrunner.iochannel) do jobschannel, exschannel, iochannel
            finish = false
            while !finish
                # Each worker takes jobs sequentially from the shared jobs pool 
                job_filename = take!(jobschannel)
                if isnothing(job_filename) # At the end there are should be only `emptyjobs`, in which case the worker finishes its tasks
                    finish = true
                else # Otherwise we assume that the `job` contains the valid `filename` and execute test
                    try # Here we can easily get the `LoadError` if some tests are failing
                        include(job_filename)
                    catch iexception
                        put!(exschannel, iexception)
                    end
                    # After the work is done we put the worker's `id` into `iochannel` (this triggers test info printing)
                    put!(iochannel, myid())
                end
            end
            return nothing
        end
        # We save the created task for later syncronization
        push!(testrunner.test_tasks, task)
    end

    # For each remotelly called task we `fetch` its result or save an exception
    foreach(fetch, testrunner.test_tasks)

    # If exception are not empty we notify the user and force-fail
    if isready(testrunner.exschannel)
        println(stderr, "Tests have failed with the following exceptions: ")
        while isready(testrunner.exschannel)
            exception = take!(testrunner.exschannel)
            showerror(stderr, exception)
            println(stderr, "\n", "="^80)
        end
        exit(-1)
    end

    close(testrunner.iochannel)
    close(testrunner.exschannel)
    close(testrunner.jobschannel)

    # At the very last stage we check that there are no "missing" tests, 
    # aka tests that have been specified in the `enabled_tests`, 
    # but for which the corresponding `filename` does not exist in the `test/` folder
    notfound_tests = filter(v -> v[2] === false, testrunner.found_tests)
    if !isempty(notfound_tests)
        println(stderr, "There are missing tests, double check correct spelling/path for the following entries:")
        foreach(keys(notfound_tests)) do key
            println(stderr, " - ", key)
        end
        exit(-1)
    end
end

const testrunner = TestRunner(lowercase.(ARGS))

println("`TestRunner` has been created. The number of available procs is $(nprocs()).")

@everywhere workerlocal_lock = ReentrantLock()

function addtests(testrunner::TestRunner, filename)
    # First we transform filename into `key` and check if we have this entry in the `enabled_tests` (if `enabled_tests` is not empty)
    key = filename_to_key(filename)
    if isempty(testrunner.enabled_tests) || key in testrunner.enabled_tests
        # If `enabled_tests` is not empty we mark the corresponding key with the `true` value to indicate that we found the corresponding `file` in the `/test` folder
        if !isempty(testrunner.enabled_tests)
            setindex!(testrunner.found_tests, true, key) # Mark that test has been found
        end
        # At this stage we simply put the `filename` into the `jobschannel` that will be processed later (see the `execute` function)
        put!(testrunner.jobschannel, filename)
    end
end

function key_to_filename(key)
    splitted = split(key, ":")
    return if length(splitted) === 1
        string("test_", first(splitted), ".jl")
    else
        string(join(splitted[1:(end - 1)], "/"), "/test_", splitted[end], ".jl")
    end
end

function filename_to_key(filename)
    splitted = split(filename, "/")
    if length(splitted) === 1
        return replace(replace(first(splitted), ".jl" => ""), "test_" => "")
    else
        path, name = splitted[1:(end - 1)], splitted[end]
        return string(join(path, ":"), ":", replace(replace(name, ".jl" => ""), "test_" => ""))
    end
end

using Aqua

if isempty(testrunner.enabled_tests)
    println("Running all tests...")
    # We `pirate` `mean` methods for distributions in `Distributions.jl`
    Aqua.test_all(ReactiveMP; ambiguities = false, piracy = false)
    # doctest(ReactiveMP)
else
    println("Running specific tests:")
    foreach(testrunner.enabled_tests) do test
        println(" - ", test)
    end
end

@testset ExtendedTestSet "ReactiveMP" begin
    @testset "Testset helpers" begin
        @test key_to_filename(filename_to_key("distributions/test_normal_mean_variance.jl")) == "distributions/test_normal_mean_variance.jl"
        @test filename_to_key(key_to_filename("distributions:normal_mean_variance")) == "distributions:normal_mean_variance"
        @test key_to_filename(filename_to_key("test_message.jl")) == "test_message.jl"
        @test filename_to_key(key_to_filename("message")) == "message"
    end

    addtests(testrunner, "algebra/test_correction.jl")
    addtests(testrunner, "algebra/test_common.jl")
    addtests(testrunner, "algebra/test_permutation_matrix.jl")
    addtests(testrunner, "algebra/test_standard_basis_vector.jl")

    addtests(testrunner, "helpers/test_helpers.jl")

    addtests(testrunner, "score/test_counting.jl")

    addtests(testrunner, "test_rule.jl")
    addtests(testrunner, "test_addons.jl")

    addtests(testrunner, "approximations/test_shared.jl")
    addtests(testrunner, "approximations/test_unscented.jl")
    addtests(testrunner, "approximations/test_linearization.jl")
    addtests(testrunner, "approximations/test_cvi.jl")

    addtests(testrunner, "constraints/prod/test_prod_analytical.jl")
    addtests(testrunner, "constraints/prod/test_prod_final.jl")
    addtests(testrunner, "constraints/prod/test_prod_generic.jl")
    addtests(testrunner, "constraints/test_factorisation.jl")

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
    addtests(testrunner, "distributions/test_gamma_inverse.jl")
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
    addtests(testrunner, "distributions/test_mixture_distribution.jl")

    addtests(testrunner, "test_message.jl")

    addtests(testrunner, "variables/test_variable.jl")
    addtests(testrunner, "variables/test_constant.jl")
    addtests(testrunner, "variables/test_data.jl")
    addtests(testrunner, "variables/test_random.jl")

    addtests(testrunner, "pipeline/test_logger.jl")

    addtests(testrunner, "test_node.jl")
    addtests(testrunner, "nodes/flow/test_flow.jl")
    addtests(testrunner, "nodes/test_addition.jl")
    addtests(testrunner, "nodes/test_bifm.jl")
    addtests(testrunner, "nodes/test_bifm_helper.jl")
    addtests(testrunner, "nodes/test_gamma_inverse.jl")
    addtests(testrunner, "nodes/test_subtraction.jl")
    addtests(testrunner, "nodes/test_probit.jl")
    addtests(testrunner, "nodes/test_autoregressive.jl")
    addtests(testrunner, "nodes/test_normal_mean_precision.jl")
    addtests(testrunner, "nodes/test_normal_mean_variance.jl")
    addtests(testrunner, "nodes/test_mv_normal_mean_precision.jl")
    addtests(testrunner, "nodes/test_mv_normal_mean_scale_precision.jl")
    addtests(testrunner, "nodes/test_mv_normal_mean_covariance.jl")
    addtests(testrunner, "nodes/test_poisson.jl")
    addtests(testrunner, "nodes/test_wishart_inverse.jl")
    addtests(testrunner, "nodes/test_or.jl")
    addtests(testrunner, "nodes/test_not.jl")
    addtests(testrunner, "nodes/test_and.jl")
    addtests(testrunner, "nodes/test_implication.jl")
    addtests(testrunner, "nodes/test_uniform.jl")
    addtests(testrunner, "nodes/test_normal_mixture.jl")

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
    addtests(testrunner, "rules/normal_mixture/test_m.jl")
    addtests(testrunner, "rules/normal_mixture/test_p.jl")
    addtests(testrunner, "rules/normal_mixture/test_switch.jl")

    addtests(testrunner, "rules/subtraction/test_marginals.jl")
    addtests(testrunner, "rules/subtraction/test_in1.jl")
    addtests(testrunner, "rules/subtraction/test_in2.jl")
    addtests(testrunner, "rules/subtraction/test_out.jl")

    addtests(testrunner, "rules/bernoulli/test_out.jl")
    addtests(testrunner, "rules/bernoulli/test_p.jl")
    addtests(testrunner, "rules/bernoulli/test_marginals.jl")

    addtests(testrunner, "rules/beta/test_out.jl")
    addtests(testrunner, "rules/beta/test_marginals.jl")

    addtests(testrunner, "rules/categorical/test_out.jl")
    addtests(testrunner, "rules/categorical/test_p.jl")
    addtests(testrunner, "rules/categorical/test_marginals.jl")

    addtests(testrunner, "rules/delta/unscented/test_out.jl")
    addtests(testrunner, "rules/delta/unscented/test_in.jl")
    addtests(testrunner, "rules/delta/unscented/test_marginals.jl")

    addtests(testrunner, "rules/delta/linearization/test_out.jl")
    addtests(testrunner, "rules/delta/linearization/test_in.jl")
    addtests(testrunner, "rules/delta/linearization/test_marginals.jl")

    addtests(testrunner, "rules/delta/cvi/test_in.jl")
    addtests(testrunner, "rules/delta/cvi/test_marginals.jl")
    addtests(testrunner, "rules/delta/cvi/test_out.jl")

    addtests(testrunner, "rules/dirichlet/test_marginals.jl")
    addtests(testrunner, "rules/dirichlet/test_out.jl")

    addtests(testrunner, "rules/dot_product/test_out.jl")
    addtests(testrunner, "rules/dot_product/test_in1.jl")
    addtests(testrunner, "rules/dot_product/test_in2.jl")
    addtests(testrunner, "rules/dot_product/test_marginals.jl")

    addtests(testrunner, "rules/softdot/test_y.jl")
    addtests(testrunner, "rules/softdot/test_theta.jl")
    addtests(testrunner, "rules/softdot/test_x.jl")
    addtests(testrunner, "rules/softdot/test_gamma.jl")

    addtests(testrunner, "rules/gamma_inverse/test_marginals.jl")
    addtests(testrunner, "rules/gamma_inverse/test_out.jl")

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

    addtests(testrunner, "rules/mv_normal_mean_scale_precision/test_out.jl")
    addtests(testrunner, "rules/mv_normal_mean_scale_precision/test_mean.jl")
    addtests(testrunner, "rules/mv_normal_mean_scale_precision/test_precision.jl")

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

    addtests(testrunner, "rules/transition/test_out.jl")
    addtests(testrunner, "rules/transition/test_a.jl")
    addtests(testrunner, "rules/transition/test_in.jl")

    run(testrunner)
end
