SHELL = /bin/bash
.DEFAULT_GOAL = help

.PHONY: help scripts_init scripts_update format check-format

# `scripts_init` only instantiates `scripts/Manifest.toml`, so every `make format` and
# `make check-format` runs the same pinned Runic; `make scripts_update` bumps it deliberately.
scripts_init:
	julia --project=scripts/ -e 'using Pkg; Pkg.instantiate(); Pkg.precompile();'

scripts_update: ## Re-resolve scripts/Manifest.toml (bumps Runic within its compat bound)
	julia --project=scripts/ -e 'using Pkg; Pkg.update(); Pkg.precompile();'

format: scripts_init ## Format Julia code
	julia --project=scripts/ scripts/formatter.jl --overwrite

check-format: scripts_init ## Check Julia code formatting (does not modify files)
	julia --project=scripts/ scripts/formatter.jl

# Every package has its own documentation site. ReactiveMP's links to the rule packages' sites
# through their published inventories.
.PHONY: doc_init docs

doc_init:
	julia --project=docs -e 'using Pkg; Pkg.instantiate();'

docs: doc_init ## Build ReactiveMP's documentation site, running its doctests
	julia --startup-file=no --project=docs docs/make.jl

.PHONY: test test-all test-reactant

test: ## Run the root suite except items tagged `:slow`. test_args="nodes", "tag:engine", "name:MessageMapping" all work; RUN_AQUA=false skips the slow Aqua checks
	julia -e 'import Pkg; Pkg.activate("."); Pkg.test(test_args = split("$(test_args)") .|> string)'

test-all: ## Run the root suite, `:slow` items included, as CI does
	TEST_ALL=true julia -e 'import Pkg; Pkg.activate("."); Pkg.test(test_args = split("$(test_args)") .|> string)'

test-reactant: ## Compile an unrolled model with Reactant (ReactiveMPReactantExt), from test/reactant's own environment
	julia --startup-file=no --project=test/reactant -e 'using Pkg; Pkg.instantiate()'
	julia --startup-file=no --project=test/reactant test/reactant/runtests.jl

help:  ## Display this help
	@awk 'BEGIN {FS = ":.*##"; printf "\nUsage:\n  make \033[36m<target>\033[0m\n"} /^[a-zA-Z_-]+:.*?##/ { printf "  \033[36m%-24s\033[0m %s\n", $$1, $$2 } /^##@/ { printf "\n\033[1m%s\033[0m\n", substr($$0, 5) } ' $(MAKEFILE_LIST)