---
title: 'ReactiveMP.jl: A Julia Package for Reactive Message Passing-based Bayesian Inference'
tags:
  - Bayesian Inference
  - Julia
  - Mesasge Passing
  - Reactive Programing
  - Variational Inference
  - galactic dynamics
  - milky way
authors:
  - name: Dmitry Bagaev # note this makes a footnote saying 'co-first author'
    orcid: 0000-0001-9655-7986
    affiliation: 1 # (Multiple affiliations must be quoted)
  - name: Bert de Vries # note this makes a footnote saying 'co-first author'
    affiliation: 1
affiliations:
 - name: Electrical Eng. Dept., Eindhoven Univ. of Technology, Eindhoven, The Netherlands
   index: 1
date: 16 September 2021
bibliography: paper.bib
---

\ 

Bayesian inference is one of the key computational mechanisms that underlies probabilistic model-based machine learning applications. Unfortunately, for many practical models, Bayesian inference requires evaluating high-dimensional integrals that have no analytical solution. As a result, Probabilistic Programming (PP) tools for Automated Approximate Bayesian Inference (AABI) have become popular, e.g., *Turing.jl* [@ge2018t], *ForneyLab.jl* [@ForneyLab.jl-2019] and others. These tools help researchers to specify probabilistic models in a high-level domain-specific language and run AABI algorithms with minimal additional overhead. 

We present *ReactiveMP.jl* package, which is a native Julia package for automated *reactive* message passing-based (both exact and approximate) Bayesian inference and Constrained Bethe Free Energy (CFBE) optimisation [@senoz_local_constraint_2021]. New package scales comfortably to inference tasks on factor graphs with tens of thousands of variables and millions of nodes. The package comes with a collection of standard probabilistic models, including linear Gaussian state-space models, hidden Markov models, auto-regressive models and mixture models. Moreover, ReactiveMP.jl API supports various processing modes such as offline learning, filtering of infinite data streams and protocols for handling missing data.

ReactiveMP.jl provides an easy way to add new models, node functions and analytical message update rules to the existing platform. The resulting inference procedures are differentiable with the *ForwardDiff.jl* [@RevelsLubinPapamarkou2016]. As for computation time and memory usage, specifically for conjugate models, the ReactiveMP.jl outperforms Turing.jl and ForneyLab.jl significantly by orders of magnitude. Performance benchmarks are available at the [GitHub repository](https://github.com/biaslab/ReactiveMP.jl).

<!-- # Statement of need

`Gala` is an Astropy-affiliated Python package for galactic dynamics. Python
enables wrapping low-level languages (e.g., C) for speed without losing
flexibility or ease-of-use in the user-interface. The API for `Gala` was
designed to provide a class-based and user-friendly interface to fast (C or
Cython-optimized) implementations of common operations such as gravitational
potential and force evaluation, orbit integration, dynamical transformations,
and chaos indicators for nonlinear dynamics. `Gala` also relies heavily on and
interfaces well with the implementations of physical units and astronomical
coordinate systems in the `Astropy` package [@astropy] (`astropy.units` and
`astropy.coordinates`).

`Gala` was designed to be used by both astronomical researchers and by
students in courses on gravitational dynamics or astronomy. It has already been
used in a number of scientific publications [@Pearson:2017] and has also been
used in graduate courses on Galactic dynamics to, e.g., provide interactive
visualizations of textbook material [@Binney:2008]. The combination of speed,
design, and support for Astropy functionality in `Gala` will enable exciting
scientific explorations of forthcoming data releases from the *Gaia* mission
[@gaia] by students and experts alike.

# Mathematics

Single dollars ($) are required for inline mathematics e.g. $f(x) = e^{\pi/x}$

Double dollars make self-standing equations:

$$\Theta(x) = \left\{\begin{array}{l}
0\textrm{ if } x < 0\cr
1\textrm{ else}
\end{array}\right.$$

You can also use plain \LaTeX for equations
\begin{equation}\label{eq:fourier}
\hat f(\omega) = \int_{-\infty}^{\infty} f(x) e^{i\omega x} dx
\end{equation}
and refer to \autoref{eq:fourier} from text.

# Citations

Citations to entries in paper.bib should be in
[rMarkdown](http://rmarkdown.rstudio.com/authoring_bibliographies_and_citations.html)
format.

If you want to cite a software repository URL (e.g. something on GitHub without a preferred
citation) then you can do it with the example BibTeX entry below for @fidgit.

For a quick reference, the following citation commands can be used:
- `@author:2001`  ->  "Author et al. (2001)"
- `[@author:2001]` -> "(Author et al., 2001)"
- `[@author1:2001; @author2:2001]` -> "(Author1 et al., 2001; Author2 et al., 2002)"

# Figures

Figures can be included like this:
![Caption for example figure.\label{fig:example}](figure.png)
and referenced from text using \autoref{fig:example}.

Figure sizes can be customized by adding an optional second parameter:
![Caption for example figure.](figure.png){ width=20% } -->

**Acknowledgements**. We acknowledge contributions from Albert Podusenko, Ismail Senoz, and Bart van Erp, and support from the whole BIASlab group during this project.

# References