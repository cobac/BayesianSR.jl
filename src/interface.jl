function get_function(sample::Sample, grammar::Grammar; latex::Bool = false)
    ex = Expr(:call, :+, sample.β[1])
    for i in eachindex(sample.trees)
        eq = Expr(:call, :*, sample.β[i + 1], get_executable(sample.trees[i], grammar))
        push!(ex.args, eq)
    end 
    expand_linear_operators!(ex)
    ex = simplify(eval(ex))
    latex && return latexify(ex)
    return ex
end 

get_function(chain::Chain; latex::Bool = false) = get_function(chain.samples[end],
                                                               chain.grammar;
                                                               latex = latex)
