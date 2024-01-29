using Aqua, CpuId, ReTestItems, ReactiveMP

Aqua.test_all(ReactiveMP; ambiguities = false, piracies = false, deps_compat = (; check_extras = false, check_weakdeps = true))

nthreads = max(cputhreads(), 1)
ncores = max(cpucores(), 1)

runtests(ReactiveMP; nworkers = ncores, nworker_threads = Int(nthreads / ncores), memory_threshold = 1.0)

