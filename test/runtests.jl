using Aqua, ReTestItems, ReactiveMP, Hwloc

Aqua.test_all(ReactiveMP; ambiguities = false, piracies = false, deps_compat = (; check_extras = false, check_weakdeps = true))

nthreads, ncores = Hwloc.num_virtual_cores(), Hwloc.num_physical_cores()
nthreads, ncores = max(nthreads, 1), max(ncores, 1)

runtests(ReactiveMP; nworkers = ncores, nworker_threads = Int(nthreads / ncores), memory_threshold = 1.0)
