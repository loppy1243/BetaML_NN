export @modelfunc, @λ, fcat, xrel, yrel

import Plots

julienne(A, dims) = mapslices(x -> [x], A, dims)
julienne(f, A, dims) = mapslices(x -> [f(x)], A, dims)

macro modelfunc(funcdecl)
    @assert funcdecl.head == :function || funcdecl.head == :(=) #=
         =# && funcdecl.args[1].head == :call

    fname = esc(funcdecl.args[1].args[1])

    quote
        $fname(modelfile::AbstractString, events, cells; model_name="") =
            $fname(BetaML_NN.load(modelfile)[:model], events, cells, model_name=model_name)
        $fname(models, events, cells; model_names=fill("", length(model_pairs))) =
            cells((m, n) -> $fname(m, events, cells, model_name=n), models, model_names)

        $(esc(funcdecl))
    end
end

macro try_defconst(expr)
    isdefined(expr.args[1]) ? nothing : Expr(:const, esc(expr))
end

_λ!(expr, splat_sym::Symbol) = ([], false)
function _λ!(expr::Expr, splat_sym::Symbol)
    syms = []
    splat = false

    map!(expr.args, expr.args) do x
        if x == :_ 
            _sym = gensym()
            push!(syms, _sym)
            _sym
        elseif x == :___
            splat = true
            Expr(:..., splat_sym)
        else
            expr.head == :call ? :(@λ $x) : x
        end
    end
    
    if expr.head != :call for x in expr.args
        sym_vec, s = _λ!(x, splat_sym)
        splat |= s
        append!(syms, sym_vec)
    end end

    (syms, splat)
end
macro λ(expr)
    splat_sym = gensym()
    syms, splat = _λ!(expr, splat_sym)

    isempty(syms) && !splat ? esc(expr) : #=
                   =# splat ? esc(:(($(syms...), $(splat_sym)...) -> $expr)) #=
                         =# : esc(:(($(syms...),) -> $expr))
end

fcat(f1, fs...) = (xs...) -> map(f -> f(xs...), (f1, fs...))
fcat(itr) = (xs...) -> map(f -> f(xs...), collect(itr))

xrel(f) = xrel(Plots.current(), f)
yrel(f) = yrel(Plots.current(), f)
xrel(plt, f) = Plots.xmin(plt) + f*(Plots.xmax(plt)-Plots.xmin(plt))
yrel(plt, f) = Plots.ymin(plt) + f*(Plots.ymin(plt)-Plots.ymax(plt))
