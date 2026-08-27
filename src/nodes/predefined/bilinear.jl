"""
Bilinear node representing the stochastic bilinear interaction
\phi(out, in, a) = exp(out a in)
"""
@node Stochastic Bilinear [out, in, a]

@average_energy Bilinear (q_out_in::Any, q_a::Any) = ...