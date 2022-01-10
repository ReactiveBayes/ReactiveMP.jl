module ReactiveMPMessageTest

using Test
using ReactiveMP 
using Distributions
using Random

import InteractiveUtils: methodswith
import Base: methods
import Base.Iterators: repeated, product
import ReactiveMP: materialize!
import ReactiveMP: mirrorlog, xtlog
import SpecialFunctions: loggamma

@testset "Message" begin

    @testset "Default methods" begin 
        data    = PointMass(1)

        for clamped in [ true, false ], initial in [ true, false ]
            msg = Message(data, clamped, initial)
            @test getdata(msg)    === data
            @test is_clamped(msg) === clamped
            @test is_initial(msg) === initial
            @test materialize!(msg) === msg
            @test occursin("Message", repr(msg))
        end

    end
    
    @testset "multiply_messages" begin 
        dist1 = NormalMeanVariance(randn(), rand())
        dist2 = NormalMeanVariance(randn(), rand())

        @test getdata(Message(dist1, false, false) * Message(dist2, false, false))    == prod(ProdAnalytical(), dist1, dist2)
        @test getdata(Message(dist2, false, false) * Message(dist1, false, false))    == prod(ProdAnalytical(), dist2, dist1)

        for (left_is_initial, right_is_initial) in product(repeated([ true, false ], 2)...)
            @test is_clamped(Message(dist1, true, left_is_initial) * Message(dist2, false, right_is_initial)) == false
            @test is_clamped(Message(dist1, false, left_is_initial) * Message(dist2, true, right_is_initial)) == false
            @test is_clamped(Message(dist1, true, left_is_initial) * Message(dist2, true, right_is_initial))  == true
            @test is_clamped(Message(dist2, true, left_is_initial) * Message(dist1, false, right_is_initial)) == false
            @test is_clamped(Message(dist2, false, left_is_initial) * Message(dist1, true, right_is_initial)) == false
            @test is_clamped(Message(dist2, true, left_is_initial) * Message(dist1, true, right_is_initial))  == true
        end

        for (left_is_clamped, right_is_clamped) in product(repeated([ true, false ], 2)...)
            @test is_initial(Message(dist1, left_is_clamped, true) * Message(dist2, right_is_clamped, true)) == !(left_is_clamped && right_is_clamped)
            @test is_initial(Message(dist2, left_is_clamped, true) * Message(dist1, right_is_clamped, true)) == !(left_is_clamped && right_is_clamped)
            @test is_initial(Message(dist1, left_is_clamped, false) * Message(dist2, right_is_clamped, false)) == false
            @test is_initial(Message(dist2, left_is_clamped, false) * Message(dist1, right_is_clamped, false)) == false    
        end

        @test is_initial(Message(dist1, true, true) * Message(dist2, true, true)) == false
        @test is_initial(Message(dist1, true, true) * Message(dist2, true, false)) == false
        @test is_initial(Message(dist1, true, false) * Message(dist2, true, true)) == false
        @test is_initial(Message(dist1, false, true) * Message(dist2, true, false)) == true
        @test is_initial(Message(dist1, true, false) * Message(dist2, false, true)) == true
        @test is_initial(Message(dist2, true, true) * Message(dist1, true, true)) == false
        @test is_initial(Message(dist2, true, true) * Message(dist1, true, false)) == false
        @test is_initial(Message(dist2, true, false) * Message(dist1, true, true)) == false
        @test is_initial(Message(dist2, false, true) * Message(dist1, true, false)) == true
        @test is_initial(Message(dist2, true, false) * Message(dist1, false, true)) == true
    end
    
    @testset "Statistics" begin 

        distributions = [ 
            PointMass(2.0),
            Gamma(10.0, 2.0), 
            NormalMeanVariance(-10.0, 10.0), 
            Wishart(4.0, [ 2.0 -0.5; -0.5 1.0 ]), 
            MvNormalMeanPrecision([ 2.0, -1.0 ], [ 7.0 -1.0; -1.0 3.0 ]), 
            Bernoulli(0.5),
            Categorical([ 0.8, 0.2 ])
        ]

        # Here we get all methods defined for a particular type of a distribution
        dists_methods = map(d -> methodswith(eval(nameof(typeof(d)))), distributions)

        methods_to_test = [
            Distributions.mean,
            Distributions.median,
            Distributions.mode,
            Distributions.shape,
            Distributions.scale,
            Distributions.rate,
            Distributions.var,
            Distributions.std,
            Distributions.cov,
            Distributions.invcov,
            Distributions.logdetcov,
            Distributions.entropy,
            Distributions.params,
            Base.precision,
            Base.length,
            Base.ndims,
            Base.size,
            mean_cov, 
            mean_invcov, 
            mean_precision, 
            weightedmean_cov, 
            weightedmean_invcov, 
            weightedmean_precision,
            probvec,
            weightedmean
        ]

        for (distribution, distribution_methods) in zip(distributions, dists_methods), method in methods_to_test
            T       = typeof(distribution)
            message = Message(distribution, false, false)
            # Here we check that a specialised method for a particular type T exist
            ms = methods(method, (T, ))
            if !isempty(ms) && all(m -> m ∈ distribution_methods, ms)
                @test method(message) == method(distribution)
            end
        end

        fn_mean_functions = (inv, log, xtlog, mirrorlog, loggamma)

        for distribution in distributions, fn_mean in fn_mean_functions
            F       = typeof(fn_mean)
            T       = typeof(distribution)
            message = Message(distribution, false, false)
            # Here we check that a specialised method for a particular type T exist
            ms = methods(mean, (F, T, ), ReactiveMP)
            if !isempty(ms)
                @test mean(fn_mean, message) == mean(fn_mean, distribution)
            end
        end

        _getpoint(rng, distritubution) = _getpoint(rng, variate_form(distritubution), distritubution)
        _getpoint(rng, ::Type{ <: Univariate }, distribution) = 10rand(rng)
        _getpoint(rng, ::Type{ <: Multivariate }, distribution) = 10 .* rand(rng, 2)

        distributions2   = [ 
            Gamma(10.0, 2.0), 
            NormalMeanVariance(-10.0, 1.0), 
            MvNormalMeanPrecision([ 2.0, -1.0 ], [ 7.0 -1.0; -1.0 3.0 ]), 
            Bernoulli(0.5),
            Categorical([ 0.8, 0.2 ])
        ]
        
        methods_to_test2 = [
            Distributions.pdf,
            Distributions.logpdf, 
        ]

        rng = MersenneTwister(1234)

        for distribution in distributions2, method in methods_to_test2
            message = Message(distribution, false, false)
            
            for _ in 1:3
                point = _getpoint(rng, distribution)
                @test method(message, point) === method(distribution, point)
            end
        end

    end

end

end