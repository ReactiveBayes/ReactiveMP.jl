using Aqua, ReTestItems, ReactiveMP, Hwloc

Aqua.test_all(ReactiveMP; ambiguities = false, piracies = false, deps_compat = (; check_extras = false, check_weakdeps = true))

nthreads, ncores = Hwloc.num_virtual_cores(), Hwloc.num_physical_cores()
nthreads, ncores = max(nthreads, 1), max(ncores, 1)
nworker_threads = Int(nthreads / ncores)
memory_threshold = 1.0

pkg_root = dirname(pathof(ReactiveMP)) |> dirname
test_root = joinpath(pkg_root, "test")

if isempty(ARGS)
    runtests(ReactiveMP; nworkers = ncores, nworker_threads = nworker_threads, memory_threshold = memory_threshold)
else
    for arg in ARGS
        # Translate colon syntax (e.g., rules:normal_mean_variance → rules/normal_mean_variance)
        candidate = join(split(arg, ":"), "/")

        # Build possible test paths relative to the package test directory
        paths = [joinpath(test_root, candidate), joinpath(test_root, candidate * ".jl")]

        path = findfirst(ispath, paths)

        if path !== nothing
            selected_path = paths[path]
            @info "Running selective tests from $selected_path"
            runtests(selected_path; nworkers = ncores, nworker_threads = nworker_threads, memory_threshold = memory_threshold)
        else
            @warn "Test target not found: $arg"
        end
    end
end
