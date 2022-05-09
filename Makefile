SHELL = /bin/bash
.DEFAULT_GOAL = help

.PHONY: lint format

lint: ## Code formating check
	julia --project=scripts/ scripts/format.jl

format: ## Code formating run
	julia --project=scripts/ scripts/format.jl --overwrite

doc_init:
	julia --project=docs -e 'ENV["PYTHON"]=""; using Pkg; Pkg.develop(PackageSpec(path=pwd())); Pkg.instantiate(); Pkg.build("PyPlot")'

.PHONY: docs

docs: ## Generate documentation
	julia --project=docs/ docs/make.jl

.PHONY: test

test: ## Run tests
	julia -e 'import Pkg; Pkg.activate("."); Pkg.test()'

help:  ## Display this help
	@awk 'BEGIN {FS = ":.*##"; printf "\nUsage:\n  make \033[36m\033[0m\n"} /^[a-zA-Z_-]+:.*?##/ { printf "  \033[36m%-24s\033[0m %s\n", $$1, $$2 } /^##@/ { printf "\n\033[1m%s\033[0m\n", substr($$0, 5) } ' $(MAKEFILE_LIST)