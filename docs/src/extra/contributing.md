# [Contributing](@id contributing)

We welcome all contributors. This page gives the guidelines to follow when contributing to this package.

## Reporting bugs

We track bugs using [GitHub issues](https://github.com/reactivebayes/ReactiveMP.jl/issues). We encourage you to write complete, specific, reproducible bug reports. Mention the versions of Julia and `ReactiveMP` for which you observe unexpected behavior. Please provide a concise description of the problem and complement it with code snippets, test cases, screenshots, tracebacks or any other information that you consider relevant. This will help us to replicate the problem and narrow the search space for solutions.

## Suggesting features

We welcome new feature proposals. However, before submitting a feature request, consider a few things:

- Does the feature require changes in the engine? A new factor node and its rules do not: they live in a rule package, in your script or notebook, or in a repository of your own (see [The ecosystem](@ref ecosystem)).
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
    [Revise.jl](https://github.com/timholy/Revise.jl) lets you modify the code and use the changes without restarting Julia.

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
- Type names use `UpperCamelCase`, such as `AbstractFactorNode` and `RandomVariable`.
- Function names are `lowercase`, as the Julia style guide has them: words run together where the name stays readable, `getdata`, `randomvar`, `factornode`, and are separated with underscores where it would not, `compute_product_of_messages`.
- Variable names and function arguments use `snake_case`
- The name of a method that modifies its argument(s) must end in `!`

### Documentation

Every package has its own documentation site under its `docs/`, built with `make docs-<package>`
(`make docs-all` builds every site in dependency order, and the ReactiveMP site last). Each site
checks that every docstring of its package appears on a page, runs its doctests, and fails on a
broken cross-reference.

A docstring is read at the REPL, on the site, and by agents working in the code, so it says
enough to use the name without reading its source:

- **The signature first**, indented, with the real argument names, keywords and defaults, and
  `-> result` when that helps. Several methods get several lines.
- **One sentence saying what it is or does**, right after: a noun phrase for a type, a value or
  an accessor ("The node's declared interfaces."), the imperative for an action ("Resolve the
  message rule towards `target` and run it.").
- **Then only the sections that carry something**, in this order: `# Arguments`, `# Keywords`
  (for each: what it accepts, its default, what it does), `# Returns`, `# Throws`,
  `# Examples`, and a final `See also` line. A simple accessor stays a line or two, as long as
  what it returns and when it fails are clear.
- **Examples** are `jldoctest`s where they are cheap and deterministic, `julia` blocks
  otherwise. Compare floating-point results with `≈`, not by their printed digits.
- **Every name that has a docstring is a link**: `` [`name`](@ref) `` within the package's own
  site, `` [`name`](@extref Package.name) `` into another package's site, such as
  `` [`DefaultAlgorithm`](@extref MessagePassingRulesBase.DefaultAlgorithm) ``. At the REPL both
  read as the plain name.
- **Text several docstrings share is written once**, as a `const` string fragment interpolated
  with `$(FRAGMENT)`, so that no docstring says "as X, except …" and the copies cannot drift.
- **State limitations plainly**: which factorisations a node supports, which message types a
  rule takes, "no average energy", "needs an initial message on `β`".
- **Present tense, British spelling** ("factorisation", "normalised"), no history ("now",
  "used to", "in v6"), no "for now" or asides. A reason is kept, in present terms.
- **Internal helpers** keep a docstring when a contributor needs it, and are listed on their
  site's *Internals* page, never among the public API.
- **Names a user is told to call are public**: exported, or declared with
  `Compat.@compat public name`, which also parses on Julia 1.10.

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

A bug in a dependency of ReactiveMP.jl is best reported in the dependency's own repository. `src/fixes.jl` holds hot-fixes until a release of the dependency fixes it.

### Makefile

`ReactiveMP.jl` uses `Makefile` for most common operations. The repository is developed and
tested on Julia 1.13.

- `make help`: Shows help snippet
- `make test`: Runs the engine's tests, except items tagged `:slow`; `make test-all` runs those too, as CI does. Both take `test_args`:
  - `make test test_args="nodes"` runs only the tests under `test/nodes/`, and `make test test_args="engine:variational"` only `test/engine/variational_tests.jl`
  - `make test test_args="tag:engine"` and `make test test_args="name:MessageMapping"` select test items by tag and by name; entries of the same kind are alternatives, entries of different kinds all apply
  - `RUN_AQUA=false make test` skips the slow Aqua checks, which are enabled by default
- `make test-<package>` runs one package's suite under `lib/`, with the same `test_args`, for example `make test-standard test_args="name:rules:Beta"`. The targets are `test-base`, `test-testutils`, `test-standard`, `test-approximations`, `test-delta`, `test-gaussian-coupling`, `test-probit`, `test-gcv`, `test-autoregressive`, `test-softdot`, `test-continuous-transition`, `test-polya`, `test-bifm`, `test-flow` and `test-discrete-transition`. They skip items tagged `:slow` unless `TEST_ALL=true` is set, and the coverage check runs only on an unfiltered run
- `make docs-<package>` builds one package's documentation site, for example `make docs-base`;
  `make docs-all` builds every site in dependency order, and `make docs` the ReactiveMP site,
  which needs the package sites built first
- `make check-format`: Checks the formatting (Runic), without modifying files
- `make format`: Formats the code; this overwrites files
- `make scripts_update`: Bumps the pinned Runic version in `scripts/Manifest.toml`
