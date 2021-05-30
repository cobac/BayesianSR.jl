# Up to 40 features
@variables begin
    x1,   x2,  x3,  x4,  x5,  x6,  x7,  x8,  x9, x10,
    x11, x12, x13, x14, x15, x16, x17, x18, x19, x20,
    x21, x22, x23, x24, x25, x26, x27, x28, x29, x30,
    x31, x32, x33, x34, x35, x36, x37, x38, x39, x40
end 

function expand_linear_operators!(ex::Expr)
    MacroTools.prewalk(ex) do x
        if isexpr(x, :call) && x.args[1] == :linear_operator
            a = x.args[2].a
            prod = Expr(:call, :*, x.args[2].b, x.args[3])
            x.args[1] = :(+)
            x.args[2] = a
            x.args[3] = prod
            return x
        else return x
        end 
    end 
end 

