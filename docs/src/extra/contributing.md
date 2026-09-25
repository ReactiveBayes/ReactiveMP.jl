# Contribution guidelines

We welcome all possible contributors. This page details the some of the guidelines that should be followed when contributing to this package.

## Reporting bugs

We track bugs using [GitHub issues](https://github.com/reactivebayes/ReactiveMP.jl/issues). We encourage you to write complete, specific, reproducible bug reports. Mention the versions of Julia and `ReactiveMP` for which you observe unexpected behavior. Please provide a concise description of the problem and complement it with code snippets, test cases, screenshots, tracebacks or any other information that you consider relevant. This will help us to replicate the problem and narrow the search space for solutions.

## Suggesting features

We welcome new feature proposals. However, before submitting a feature request, consider a few things:

- Does the feature require changes in the core ReactiveMP.jl code? If it doesn't (for example, you would like to add a factor node for a particular application), you can add local extensions in your script/notebook or consider making a separate repository for your extensions.
- If you would like to add an implementation of a feature that changes a lot in the core ReactiveMP.jl code, please open an issue on GitHub and describe your proposal first. This will allow us to discuss your proposal with you before you invest your time in implementing something that may be difficult to merge later on.

## Contributing code

### Installing ReactiveMP

We suggest that you use the `dev` command of the Julia package manager to
install ReactiveMP.jl for development purposes. To work on your fork of ReactiveMP.jl, use your fork's URL address in the `dev` command, for example:

```
] dev git@github.com:your_username/ReactiveMP.jl.git
```

The `dev` command clones ReactiveMP.jl to `~/.julia/dev/ReactiveMP`. All local
changes to ReactiveMP code will be reflected in imported code.

!!! note
    It is also might be useful to install [Revise.jl](https://github.com/timholy/Revise.jl) package as it allows you to modify code and use the changes without restarting Julia.

### Committing code

We use the standard [GitHub Flow](https://guides.github.com/introduction/flow/) workflow where all contributions are added through pull requests. In order to contribute, first [fork](https://guides.github.com/activities/forking/) the repository, then commit your contributions to your fork, and then create a pull request on the `main` branch of the ReactiveMP.jl repository.

Before opening a pull request, please make sure that all tests pass without
failing, that `make check-format` reports no changes, and that `CHANGELOG.md` has an entry
describing your change (CI enforces all three).

### Style conventions

!!! note
    ReactiveMP.jl repository contains scripts to automatically format code according to our guidelines. Use `make format` command to fix code style. This command overwrites files.

We use default [Julia style guide](https://docs.julialang.org/en/v1/manual/style-guide/index.html). We list here a few important points and our modifications to the Julia style guide:

- Use 4 spaces for indentation
- Type names use `UpperCamelCase`. For example: `AbstractFactorNode`, `RandomVariable`, etc..
- Function names are `lowercase` with underscores, when necessary. For example: `activate!`, `randomvar`, etc..
- Variable names and function arguments use `snake_case`
- The name of a method that modifies its argument(s) must end in `!`

### Unit tests

We use the test-driven development (TDD) methodology for ReactiveMP.jl development. The test coverage should be as complete as possible. Please make sure that you write tests for each piece of code that you want to add.

The engine's tests are in the `/test/` directory, which follows the structure of the `/src/` directory; each rule package under `/lib/` has its own `test/` directory. Test files are named `*_tests.jl` and hold `@testitem` blocks, each carrying a tag. A rule package's suite ends with a coverage check: every rule it defines must be selected by some test, a table case or a direct `call_*`. Some tests are also present in `jldoctest` docs annotations directly in the source code.
See [Julia's documentation](https://docs.julialang.org/en/v1/manual/documentation/index.html) about doctests.

The tests can be evaluated by running following command in the Julia REPL:

```
] test ReactiveMP
```

In addition tests can be evaluated by running following command in the ReactiveMP root directory:

```bash
make test
```

### Fixes to external libraries 

If a bug has been discovered in an external dependencies of the `ReactiveMP.jl` it is the best to open an issue 
directly in the dependency's github repository. You use can use the `fixes.jl` file for hot-fixes before 
a new release of the broken dependency is available.

### Makefile

`ReactiveMP.jl` uses `Makefile` for most common operations. The repository is developed and
tested on Julia 1.13.

- `make help`: Shows help snippet
- `make test`: Runs the engine's tests, except items tagged `:slow`; `make test-all` runs those too, as CI does. Both take `test_args`:
  - `make test test_args="nodes"` runs only the tests under `test/nodes/`, and `make test test_args="engine:variational"` only `test/engine/variational_tests.jl`
  - `make test test_args="tag:engine"` and `make test test_args="name:MessageMapping"` select test items by tag and by name; entries of the same kind are alternatives, entries of different kinds all apply
  - `RUN_AQUA=false make test` skips the slow Aqua checks, which are enabled by default
- `make test-<package>` runs one package's suite under `lib/`, with the same `test_args`, for example `make test-standard test_args="name:rules:Beta"`. The targets are `test-base`, `test-testutils`, `test-standard`, `test-approximations`, `test-delta`, `test-gaussian-coupling`, `test-probit`, `test-gcv`, `test-autoregressive`, `test-softdot`, `test-continuous-transition`, `test-polya`, `test-bifm`, `test-flow` and `test-discrete-transition`. They skip items tagged `:slow` unless `TEST_ALL=true` is set, and the coverage check runs only on an unfiltered run
- `make docs`: Builds the documentation, running its doctests
- `make check-format`: Checks the formatting (Runic), without modifying files
- `make format`: Formats the code; this overwrites files
- `make scripts_update`: Bumps the pinned Runic version in `scripts/Manifest.toml`
