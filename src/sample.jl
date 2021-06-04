
"""
    Sample(trees::Vector{RuleNode}, β::Vector{Float64}, σ²::Float64)

Each sample of a `Chain` is one equation.

- `trees`: Vector with all `RuleNode`.
- `β`: Vector with the linear coefficients.
  - `β[1]`: Intercept.
- `σ²`: Dictionary with variances
  - `:σ²`: of the residuals
  - `:σ²_a`: of the LinearCoef intercepts
  - `:σ²_b`: of the LinearCoef slopes

"""
struct Sample
    trees::Vector{RuleNode}
    β::Vector{Float64}
    σ²::Dict{Symbol,AbstractFloat}
end 

"""
    Sample(k::Real, grammar::Grammar, hyper::Hyperparams)

- `k` is the number of `RuleNode` that are added in each equation.
"""
function Sample(grammar::Grammar, hyper::Hyperparams)
    @unpack k, σ²_prior, σ²_a_prior, σ²_b_prior = hyper
    Sample([RuleNode(grammar, hyper) for _ in 1:k],
           zeros(k + 1),
           Dict([(:σ², rand(σ²_prior)),
                 (:σ²_a, rand(σ²_a_prior)),
                 (:σ²_b, rand(σ²_b_prior))]))
end 

"""
    new_sample_recursive(k::Int, grammar::Grammar, hyper::Hyperparams, x::Matrix, y::Vector)

Generates a new sample guaranteeing that the mathematical expression is well behaved (i.e. no NaNs).
"""
function new_sample_recursive(grammar::Grammar, hyper::Hyperparams, x::Matrix, y::Vector)
    sample = Sample(grammar, hyper)
    try 
        optimβ!(sample, x, y, grammar)
    catch e 
        return new_sample_recursive(grammar, hyper, x, y)
    end 
    return sample
end 
