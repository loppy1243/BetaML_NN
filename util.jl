export @modelfunc

julienne(A, dims) = mapslices(x -> [x], A, dims)
julienne(f, A, dims) = mapslices(x -> [f(x)], A, dims)

# Alternative: create zero array of correct size and fill in with A.
#function pad(A, val=0; width=1)
#    ret = A
#    dims = size(A) |> collect
#    off = zeros(Int, ndims(A))
#    for i = 1:length(dims)
#        ds = dims + off
#        zs = fill(val, ds[1:i-1]..., 1, ds[i+1:end]...)
#        for _ = 1:width
#            ret = cat(i, zs, ret)
#            ret = cat(i, ret, zs)
#        end
#        off[i] += 2width
#    end
#
#    ret
#end

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
    isdefined(expr.args[1].args[1]) ? nothing : :(const $(esc(expr.args[1])) = $expr)
end
