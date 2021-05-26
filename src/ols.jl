"""
   optimβ!(sample::Sample, x, y, grammar::Grammar) 

Optimises the β coefficients of a `Sample` by Ordinary Least Squares.

"""
function optimβ!(sample::Sample, x, y, grammar::Grammar)
    n = size(x, 1)
    xs = Matrix{Float64}(undef, n, length(sample.trees) + 1)
    xs[:, 1] = ones(size(x, 1))
    xs[:, 2:end] = evalsample(sample, x, grammar)
    # QR decomposition
    β = xs \ y
    for i in eachindex(sample.β)
        sample.β[i] = β[i]
    end 
    return nothing
end 
