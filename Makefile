SHELL = /bin/bash
.DEFAULT_GOAL = help

lint:
	julia scripts/format.jl