module ModelMacro

using ..Model

const dict = Dict()

function modeldef(name::Symbol, expr::Expr, params...)
    length(params) == 0 && return :(BetaML_NN.ModelMacro.dict[$(Meta.quot(name))] = $expr)

    @assert expr.head == :(->)
    @assert expr.args[1] == Expr(:tuple)
    @assert expr.args[2].head == :block

    fname = gensym()

    ret_expr = Expr(:block)

    for line in expr.args[2].args
        if line.head == :line
            push!(ret_expr.args, line)
            continue
        end
        @assert line.head == :(->)

        args = if line.args[1].head == :tuple
            line.args[1].args
        else
            [line.args[1]]
        end

        new_line = Expr(:(=), Expr(:call, fname, params..., args...),
                              line.args[2])
        push!(ret_expr.args, new_line)
    end

    quote
        $ret_expr
        BetaML_NN.ModelMacro.dict[$(Meta.quot(name))] = $fname
    end
end

# Pretty sure this always works...
param_name(param::Symbol) = param
param_name(param::Expr) = param_name(param.args[1])
parse_param(param::Symbol) = (param, param, nothing)
parse_param(param::Expr) = if param.head == :call && param.args[1] == :(=>)
    (param.args[2], param_name(param.args[2]), param.args[3])
else
    (param, param_name(param), nothing)
end

macro loss(expr, params...)
    esc(modeldef(:loss, expr, params...))
end
macro opt(expr, params...)
    esc(modeldef(:opt, expr, params...))
end
macro model(expr, params...)
    esc(modeldef(:model, expr, params...))
end
macro params(expr)
    esc(quote
        BetaML_NN.ModelMacro.dict[:params] = $expr
    end)
end

macro model(expr::Expr)
    @assert expr.head in (:function, :(=)) && expr.args[1].head == :call

    parsed_ps = map(parse_param, expr.args[1].args[2:end])
    ppairs = map(parsed_ps) do t
        _, pname, pdesc = t
        pdesc === nothing ? :($(Meta.quot(pname)) => $pname) : :($(Meta.quot(pname)) => $pdesc)
    end
    pdefs = map(filter(x -> x[3] !== nothing, parsed_ps)) do t
        _, pname, pdesc = t
        :($pdesc = last($pname); $pname = first($pname))
    end

    head_expr = quote; $(pdefs...) end
    ret_expr = quote
        try
            Model(BetaML_NN.ModelMacro.dict[:model], BetaML_NN.ModelMacro.dict[:loss],
                  BetaML_NN.ModelMacro.dict[:opt], BetaML_NN.ModelMacro.dict[:params],
                  $(ppairs...))
        finally
            delete!(BetaML_NN.ModelMacro.dict, :model)
            delete!(BetaML_NN.ModelMacro.dict, :loss)
            delete!(BetaML_NN.ModelMacro.dict, :opt)
            delete!(BetaML_NN.ModelMacro.dict, :params)
        end
    end

    expr.args[1].args[2:end] .= map(x->x[1], parsed_ps)
    for line in expr.args[2].args
        if line isa Expr #=
        =# && line.head == :macrocall #=
        =# && line.args[1] in map(Symbol, ("@loss", "@opt", "@model", "@params"))
            line.args[1] = :(BetaML_NN.ModelMacro.$(line.args[1]))
        end
    end
    prepend!(expr.args[2].args, [head_expr])
    push!(expr.args[2].args, ret_expr)

    esc(expr)
end

end # module ModelMacro
