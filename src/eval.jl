
"""
    evaltree(tree::RuleNode, x, grammar::Grammar)

Evaluates an `RuleNode`. Outputs a vector with a value for every observation.
"""
function evaltree(tree::RuleNode, x, grammar::Grammar)
    #= Evaluates a tree into a vector for every variable x_j =#
    n = size(x, 1)
    out = Vector{Float64}(undef, n)
    eq = get_executable(tree, grammar)
    @inbounds for i in 1:n
        table = tableforeval(x, i, grammar) 
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
    @inbounds for j in eachindex(sample.trees)
        out[:, j] = evaltree(sample.trees[j], x, grammar)
    end 
    return out
end 

"""
    tableforeval(x, i, grammar::Grammar)

Generates a symbol table to evaluate the i-th observation in a `RuleNode`.
"""
function tableforeval(x, i, grammar::Grammar)
    symboltable = SymbolTable(grammar, BayesianSR)
    @inbounds for m in axes(x, 2)
        symboltable[Symbol("x", m)] = x[i, m]
    end 
    return symboltable
end 
