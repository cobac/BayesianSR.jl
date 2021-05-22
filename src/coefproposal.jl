"""
    SetLinearCoef

Object with the set of LinearCoef of a RuleNode.
"""
struct SetLinearCoef
    a::Vector{AbstractFloat}
    b::Vector{AbstractFloat}
end 

"""
    recover_LinearCoef(tree::RuleNode)

Returns an object of type `SetLinearCoef` with the `LinearCoef` of a `RuleNode`.
"""
function recover_LinearCoef(tree::RuleNode)
    a = AbstractFloat[]
    b = AbstractFloat[]
    queue = [tree]
    while !isempty(queue)
        current = queue[1]
        if current.ind == 1 # LinearCoef
            push!(a, current._val.a)
            push!(b, current._val.b)
        end 
        append!(queue, queue[1].children)
        deleteat!(queue, 1)
    end
    return SetLinearCoef(a, b)
end 

"""
    propose_LinearCoef!(tree::RuleNode, σ²_a::AbstractFloat, σ²_b::AbstractFloat)
Updates a `RuleNode` with a new set of `LinearCoef`.
Returns an object of type `SetLinearCoef` with the new coefficients.
"""
function propose_LinearCoef!(tree::RuleNode, σ²_a::AbstractFloat, σ²_b::AbstractFloat)
    new_a = AbstractFloat[]
    new_b = AbstractFloat[]
    queue = [tree]
    while !isempty(queue)
        current = queue[1]
        if current.ind == 1 # LinearCoef
            current._val = LinearCoef(σ²_a, σ²_b, variances=true)
            push!(new_a, current._val.a)
            push!(new_b, current._val.b)
        end 
        append!(queue, queue[1].children)
        deleteat!(queue, 1)
    end
    return SetLinearCoef(new_a, new_b)
end 

"""
    propose_LinearCoef!(tree::RuleNode, σ²_a::AbstractFloat, σ²_b::AbstractFloat, 
                             θ_old::SetLinearCoef, mov::Symbol)
Updates a `RuleNode` with a new set of `LinearCoef`.
Returns an object of type `SetLinearCoef` with the new coefficients
and two `SetLinearCoef` with the auxiliary variables U and U⁺.
"""
function propose_LinearCoef!(tree::RuleNode, σ²_a::AbstractFloat, σ²_b::AbstractFloat,
                             θ_old::SetLinearCoef, mov::Symbol)
    θ_old = deepcopy(θ_old)
    new_a = AbstractFloat[]
    new_b = AbstractFloat[]
    U_a = AbstractFloat[]
    U_b = AbstractFloat[]
    U⁺_a = AbstractFloat[]
    U⁺_b = AbstractFloat[]
    queue = [tree]
    if mov == :expansion
        while !isempty(queue)
            current = queue[1]
            if current.ind == 1 # LinearCoef
                u_a = rand(Normal(1, σ²_a))
                u_b = rand(Normal(0, σ²_b))
                push!(U_a, u_a)
                push!(U_b, u_b)
                if isempty(θ_old.a)
                    current._val = LinearCoef(u_a, u_b)
                    push!(new_a, u_a)
                    push!(new_b, u_b)
                else
                    push!(U⁺_a, (θ_old.a[1] - u_a) / 2)
                    push!(U⁺_b, (θ_old.b[1] - u_b) / 2)
                    a = (u_a + current._val.a) / 2
                    b = (u_b + current._val.b) / 2
                    push!(new_a, a)
                    push!(new_b, b)
                    current._val = LinearCoef(a, b)
                    popfirst!(θ_old.a)
                    popfirst!(θ_old.b)
                end 
            end 
            append!(queue, current.children)
            deleteat!(queue, 1)
        end 
    else # mov == :shrinkage
        count = 1
        while !isempty(queue)
            current = queue[1]
            if current.ind == 1 # LinearCoef
                count += 1
                u_a = rand(Normal(0, σ²_a))
                u_b = rand(Normal(0, σ²_b))
                push!(U_a, u_a)
                push!(U_b, u_b)
                a =  u_a + current._val.a
                b =  u_b + current._val.b
                push!(new_a, a)
                push!(new_b, b)
                current._val = LinearCoef(a, b)
                θ_old.a[count] = θ_old.a[count] - u_a
                θ_old.b[count] = θ_old.b[count] - u_b
            end 
            append!(queue, current.children)
            deleteat!(queue, 1)
        end 
        U⁺_a = θ_old.a
        U⁺_b = θ_old.b
    end 
    return (SetLinearCoef(new_a, new_b),
            SetLinearCoef(U_a, U_b),
            SetLinearCoef(U⁺_a, U⁺_b))
end 

"""
    any_linear_operators(tree::RuleNode)

Returns a `Bool`.
    """
function any_linear_operators(tree::RuleNode)
        queue = [tree]
        while !isempty(queue)
            current = queue[1]
            if current.ind == 1 # LinearCoef
                return true
            end 
            append!(queue, queue[1].children)
            deleteat!(queue, 1)
        end
        return false
    end 
