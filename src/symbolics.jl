# Up to 40 features
@variables begin
    x1,   x2,  x3,  x4,  x5,  x6,  x7,  x8,  x9, x10,
    x11, x12, x13, x14, x15, x16, x17, x18, x19, x20,
    x21, x22, x23, x24, x25, x26, x27, x28, x29, x30,
    x31, x32, x33, x34, x35, x36, x37, x38, x39, x40
end 

"""
    make_readable(ex::Expr; round_digits::Bool = true)

Expands linear operators and optionally rounds all real numbers to 3 decimal digits.
"""
function make_readable(ex::Expr; round_digits::Bool=true)
    MacroTools.prewalk(ex) do x
        if isexpr(x, :call) && x.args[1] == :linear_operator
            a = x.args[2].a
            prod = Expr(:call, :*, x.args[2].b, x.args[3])
            x.args[1] = :(+)
            x.args[2] = a
            x.args[3] = prod
            return x
        elseif typeof(x) <: Number
            round_digits && return round(x, digits=3)
            return x
        else return x
        end 
    end 
end 

function rm_symbolic_clutter!(ex::Expr)
    # rm inv
    if Symbol(string(ex.args[1])) == :*
        if isexpr(ex.args[2], :call) && isexpr(ex.args[3], :call) &&
        Symbol(string(ex.args[2].args[1])) == :inv &&
        Symbol(string(ex.args[3].args[1])) == :inv 
            error()
        elseif isexpr(ex.args[2], :call) && Symbol(string(ex.args[2].args[1])) == :inv
            ex.args[1] = (:/)
            denom = ex.args[2].args[2]
            ex.args[2] = ex.args[3]
            ex.args[3] = denom
        elseif isexpr(ex.args[3], :call) && Symbol(string(ex.args[3].args[1])) == :inv
            ex.args[1] = (:/)
            ex.args[3] = ex.args[3].args[2]
        else 
        end 
        # rm ^
    elseif Symbol(string(ex.args[1])) == :^
        if ex.args[3] == 2 # x^2
            ex.args[1] = (:*)
            ex.args[3] = ex.args[2]
        elseif ex.args[3] == 3 # x^3
            ex.args[1] = (:*)
            ex.args[3] = ex.args[2]
            push!(ex.args, ex.args[2])
        end 
    end 
    return ex
end 

rm_symbolic_clutter!(s::Symbol) = s

"""
    symbolic_to_RuleNode(eq::Expr, g::Grammar)

Transforms a symbolic equation to a RuleNode.
"""
function symbolic_to_RuleNode(eq::Expr, g::Grammar)
    rm_symbolic_clutter!(eq)
    len = length(eq.args)
    main = Symbol(string(eq.args[1]))
    operations = rule_symbols(g)
    operator_is = operator_indices(g)
    node = RuleNode(0)
    if len == 1
        # Should only be called in the case where the whole equation is just a terminal
        node.ind = symbolic_to_RuleNode(eq.args[1], g)
    elseif len > 3 
        if !in(main, operations)
            error("Symbol not found in grammar (+3): ", main)
        end 
        rest = Expr(:call, main, Tuple(eq.args[3:end])...)
        ex = Expr(:call, main, eq.args[2], rest)
        ex = rm_symbolic_clutter!(ex)
        node = symbolic_to_RuleNode(ex, g)
        # a + bx
    elseif main == :+ &&
        isa(eq.args[2], Number) &&
        isexpr(eq.args[3], :call) &&
        Symbol(string(eq.args[3].args[1])) == :* &&
        isa(eq.args[3].args[2], Number)
        # eq = rm_symbolic_clutter!(eq)
        node.ind = 2
        push!(node.children, RuleNode(1, LinearCoef(eq.args[2], eq.args[3].args[2])),
              symbolic_to_RuleNode(eq.args[3].args[3], g))
        # a + x
    elseif main == :+ && isa(eq.args[2], Number)
        # eq = rm_symbolic_clutter!(eq)
        node.ind = 2
        push!(node.children, RuleNode(1, LinearCoef(eq.args[2], 1)),
              symbolic_to_RuleNode(eq.args[3], g))
        # bx
    elseif main == :* && isa(eq.args[2], Number)
        # eq = rm_symbolic_clutter!(eq)
        node.ind = 2
        push!(node.children, RuleNode(1, LinearCoef(0, eq.args[2])),
              symbolic_to_RuleNode(eq.args[3], g))
    else 
        # eq = rm_symbolic_clutter!(eq)
        if !in(main, operations)
            error("Symbol not found in grammar: ", main)
        end 
        i = findall(x -> x == main, operations)[1]
        node.ind = operator_is[i]
        for child in 2:len
            push!(node.children, symbolic_to_RuleNode(eq.args[child], g))
        end 
    end 
    return node
end 

function symbolic_to_RuleNode(eq::Num, g::Grammar)
    symbolic_to_RuleNode(Symbolics.toexpr(eq), g)
end 

function symbolic_to_RuleNode(s::Symbol, grammar::Grammar)
    i = parse(Int, string(s)[2:end])
    terminal_is = terminal_indices(grammar)
    return RuleNode(terminal_is[i])
end 

"""
    get_function(sample::Sample, grammar::Grammar; latex::Bool = false)

Extracts the symbolic representation of a `Sample`Sample`
"""
function get_function(sample::Sample, grammar::Grammar; latex::Bool=false)
    ex = Expr(:call, :+, sample.β[1])
    for i in eachindex(sample.trees)
        eq = Expr(:call, :*, sample.β[i + 1], get_executable(sample.trees[i], grammar))
        push!(ex.args, eq)
    end 
    ex = eval(make_readable(ex))
    latex && return latexify(ex)
    return ex
end 

"""
    get_function(chain::Chain; latex::Bool = false)

Extracts the symbolic representation of the last `Sample` of a `Chain`.
"""
get_function(chain::Chain; latex::Bool=false) = get_function(chain.samples[end],
                                                             chain.grammar;
                                                             latex=latex)

"""
    get_function(tree::RuleNode, grammar::Grammar)

Extracts the symbolic representation of a tree represented by a `RuleNode`.
"""
function get_function(tree::RuleNode, grammar::Grammar)
    return eval(make_readable(get_executable(tree, grammar), round_digits=false))
end 
