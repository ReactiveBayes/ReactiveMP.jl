@testitem "quality:doctests" tags = [:quality] begin
    using Documenter, ReactiveMP, BayesBase, Distributions, ExponentialFamily

    # The engine's docstring examples run with the tests, as each lib package's do; the docs
    # build runs them again with the pages.
    DocMeta.setdocmeta!(ReactiveMP, :DocTestSetup, :(using ReactiveMP, BayesBase, Distributions, ExponentialFamily); recursive = true)
    doctest(ReactiveMP; manual = false)
end
