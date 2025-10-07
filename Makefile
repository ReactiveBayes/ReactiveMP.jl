SHELL = /bin/bash
.DEFAULT_GOAL = help

.PHONY: lint format

scripts_init:
	julia --project=scripts/ -e 'using Pkg; Pkg.instantiate(); Pkg.update(); Pkg.precompile();'

format: scripts_init ## Format Julia code
	julia --project=scripts/ scripts/formatter.jl --overwrite

check-format: scripts_init ## Check Julia code formatting (does not modify files)
	julia --project=scripts/ scripts/formatter.jl

.PHONY: benchmark

bench: ## Run benchmark, use `make bench branch=...` to test against a specific branch
	julia --startup-file=no --project=scripts/ scripts/bench.jl $(branch)

.PHONY: docs

doc_init:
	julia --project=docs -e 'using Pkg; Pkg.develop(PackageSpec(path=pwd())); Pkg.instantiate();'

docs: doc_init ## Generate documentation
	julia --project=docs/ docs/make.jl

.PHONY: test

test: ## Run tests
	julia -e 'import Pkg; Pkg.activate("."); Pkg.test()'	
	
help:  ## Display this help
	@awk 'BEGIN {FS = ":.*##"; printf "\nUsage:\n  make \033[36m\033[0m\n"} /^[a-zA-Z_-]+:.*?##/ { printf "  \033[36m%-24s\033[0m %s\n", $$1, $$2 } /^##@/ { printf "\n\033[1m%s\033[0m\n", substr($$0, 5) } ' $(MAKEFILE_LIST)