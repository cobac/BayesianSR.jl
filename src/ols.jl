"""
   optimβ!(sample::Sample, x, y, grammar::Grammar) 

Optimises the β coefficients of a `Sample` by Ordinary Least Squares.

See also: `ols`
"""
function optimβ!(sample::Sample, x, y, grammar::Grammar)
    n = size(x, 1)
    xs = Matrix{Float64}(undef, n, length(sample.trees))
    for k in eachindex(sample.trees)
        xs[:, k] = evaltree(sample.trees[k], x, grammar)
    end 
    β = ols(y, xs)
    for i in eachindex(sample.β)
        sample.β[i] = β[i]
    end 
    return nothing
end 

"""
    ols(y, x)

Ordinary Least Squares via QR decomposition.
"""
function ols(y, x)
    X = [ones(size(x, 1)) x]
    β = X \ y
    return β
end 
