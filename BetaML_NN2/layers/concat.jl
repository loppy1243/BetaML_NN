module ConcatLayer
export @Concat

import Flux

struct Concat
    components::Vector
    argmap::Vector{Tuple{Union{Colon, Vector{Int}}, Bool}}

    function Concat(args::Vararg{Tuple{Any, Union{Colon, Vector{Int}}, Bool}})
        components = map(x -> x[1], args) |> collect
        argmap = map(x -> x[2:3], args) |> collect

        new(components, argmap)
    end
end
Flux.params(c::Concat) = reduce(vcat, map(Flux.params, c.components))
function (c::Concat)(x)
    ret = []
    for (c, ixs_splat) in zip(c.components, c.argmap)
        ixs, splat = ixs_splat
        
        args = if ixs === Colon()
            x
        elseif length(ixs) == 1
            x[ixs[1]]
        else
            x[ixs]
        end
            
        if splat
            for y in c(args)
                push!(ret, y)
            end
        else
            push!(ret, c(args))
        end
    end

    ret
end

macro Concat(xs...)
    ret = Expr(:call, :Concat)
    for x in xs
        push!(ret.args,
              if x isa Expr
                  if x.head === :call && x.args[1] === :(=>)
                      if x.args[3] isa Expr && x.args[3].head === :...
                          Expr(:tuple, esc(x.args[3].args[1]), :(vec(collect($(esc(x.args[2]))))), true)
                      else
                          Expr(:tuple, esc(x.args[3]), :(vec(collect($(esc(x.args[2]))))), false)
                      end
                  elseif x.head === :...
                      Expr(:tuple, esc(x.args[1]), :, true)
                  else
                      Expr(:tuple, esc(x), :, false)
                  end
              else
                  Expr(:tuple, esc(x), :, false)
              end)
    end

    ret
end

end # module ConcatLayer
