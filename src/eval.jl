
"""
    evaltree(tree::RuleNode, x, grammar::Grammar)

Evaluates an `RuleNode`. Outputs a vector with a value for every observation.
"""
function evaltree(tree::RuleNode, x, grammar::Grammar, table::Dict{Symbol, Any})
    n = size(x, 1)
    out = Vector{Float64}(undef, n)
    eq = get_executable(tree, grammar)
    @inbounds for i in 1:n
        tableforeval!(table, x, i) 
        out[i] = Core.eval(table, eq)
    end 
    return out
end 

"""
    evalsample(sample::Sample, x, grammar::Grammar)

Evaluates a `Sample`. Outputs a matrix with the evaluations of every `RuleNode`.
"""
function evalsample(sample::Sample, x, grammar::Grammar)
    out = Matrix{Float64}(undef, size(x, 1), length(sample.trees))
    table = SymbolTable(grammar, BayesianSR)
    @inbounds for j in eachindex(sample.trees)
        out[:, j] = evaltree(sample.trees[j], x, grammar, table)
    end 
    return out
end 

"""
    tableforeval!(table::Dict{Symbol, Any}, x, i)

Updates a symbol table to evaluate the i-th observation in a `RuleNode`.
"""
function tableforeval!(table::Dict{Symbol, Any}, x, i)
    @inbounds for m in axes(x, 2)
        table[Symbol("x", m)] = x[i, m]
    end 
    return nothing
end 

function evalmodel(sample::Sample, x, grammar::Grammar)
    return sample.β[begin] .+ evalsample(sample, x, grammar) * view(sample.β, 2:length(sample.β))
end 
