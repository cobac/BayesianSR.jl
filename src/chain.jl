"""
    Chain

A chain of samples from the posterior space.

- `samples`: Vector will all the samples.
- `grammar`: The complete `Grammar` with all operators and features.
- `x`: Matrix of the features.
- `y`: Vector of the outcome values.
- `stats`: `Dict` with statistics about the `Chain`.
  - `:lastj`: Index of the last `RuleNode` that was sampled during MCMC.
- `hyper`: `Hyperparameters` of the `Chain`.

"""
struct Chain
    samples::Vector{Sample}
    grammar::Grammar
    x::Matrix{Float64}
    y::Vector{Float64}
    stats::Dict{Symbol, Int}
    hyper::Hyperparams
end 

"""
    Chain(x::Matrix, y::Vector; operators::Grammar = deepcopy(defaultgrammar), hyper::Hyperparams = Hyperparams())

Initialize a `Chain`.
"""
function Chain(x::Matrix, y::Vector;
               operators::Grammar = deepcopy(defaultgrammar),
               hyper::Hyperparams = Hyperparams())
    @unpack k, σ²_prior = hyper
    grammar = append!(deepcopy(lineargrammar),
                      append!(deepcopy(operators), variablestogrammar(x)))
    sample = Sample(k, grammar, hyper)
    try 
        optimβ!(sample, x, y, grammar)
    catch e 
        sample.β = zeros(k+1)
    end 
    stats = Dict([(:lastj, 0)])
    return Chain([sample], grammar, x, y, stats, hyper)
end 

"""
    length(chain::Chain) = length(chain.samples)

Number of samples in a `Chain`.
"""
Base.length(chain::Chain) = length(chain.samples)

"""
    no_trees(chain::Chain) = length(chain.samples[1].trees)

Number of `RuleNode` per `Sample` of a `Chain`.
"""
no_trees(chain::Chain) = length(chain.samples[1].trees)
