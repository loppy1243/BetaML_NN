export @λ, fcat, xrel, yrel

import Plots

julienne(A, dims) = mapslices(x -> [x], A, dims)
julienne(f, A, dims) = mapslices(x -> [f(x)], A, dims)

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
function xrel(plt, f)
    xmin, xmax = Plots.xlims(plt)
    xmin + f*(xmax-xmin)
end
function yrel(plt, f)
    ymin, ymax = Plots.ylims(plt)
    ymin + f*(ymax-ymin)
end
