export @modelfunc

using Plots
using StatsBase: mode

## TODO: Dream
#@def_functype modelfunc begin
#    modelfunc(modelfile::AbstractString events, cells, args...; kws...) =
#        modelfunc(BetaML_NN.load(modelfile), events, cells, args...; kws...)
#end
#macro def_functype(name, expr)
#    quote
#        macro $(esc(name))(funcdecl)
#            @assert funcdecl.head == :function || funcdecl.head == :(=) #=
#                 =# && funcdecl.args[1].head == :call
#
#            fname = esc(funcdecl.args[1].args[1])
#
#            quote
#            end
#        end
#    end
#end

macro modelfunc(funcname)
    fname = esc(funcname)

    quote
        $fname(modelfile::AbstractString, events, cells, args...; kws...) =
            $fname(BetaML_NN.load(modelfile), events, cells, args...; kws...)
    end
end

@modelfunc pointhist
pointhist(model, args...; kws...) = pointhist([model], args...; kws...)
function pointhist(models::AbstractVector, events, points, y, p; model_name=[], color=[])
    print("Computing predictions...")
    preds = map(x -> predict(x, events), models)
    println(" Done.")
  
    print("Computing distances...")
    dists = map(x -> squeeze(mapslices(norm, x - points, 1), 1), preds)
    println(" Done.")

    print("Generating histogram...")

    cs = @reshape color[_, :]
    linhist1 = stephist(dists, label=model_name, color=cs)
    linhist2 = stephist(dists, legend=false, xlims=(0, 4), color=cs)
    loghist1 = stephist(dists, legend=false, yaxis=(:log10, (1, Inf)), color=cs,
                        xlabel="Dist. of pred. from true (mm)")

    blank = plot(axis=false)
    hist = plot(layout=(1, 2), plot(layout=(2, 1), linhist1, loghist1),
                               plot(layout=(2, 1), linhist2, blank),
                               size=(3*500, 500))
    println(" Done.")

    print("Computing statistics...") 
    ms = map(minimum, dists)
    Ms = map(maximum, dists)
    mes = map(mean, dists)
    mos = map(mode, dists)
    sts = map((x, y) -> std(x, mean=y), dists, mes)
    xs = map(dists) do ds
        count(ds .< y)/length(ds)
    end
    qs = map(dists) do ds
        Base.quantile(ds, p)
    end
    println(" Done.")

    for (name, stats) in zip(model_name, zip(xs, qs, ms, Ms, mes, mos, sts))
        padding = isempty(model_name) ? "" : " "^4
        !isempty(model_name) && println("$name:")
        map(("P(<$y)", "$(p.*100)-%tiles", "Min", "Max", "Mean", "Mode", "SD"), stats) do a, b
            println(padding, a, " = ", b)
        end
    end

    (hist, xs, qs, ms, Ms, mes, mos, sts)
end

@modelfunc quantile
function quantile(model, events, points, y, p)
    print("Computing predictions...")
    pred_points = predict(model, events)
    println(" Done.")

    print("Computing distances...")
    dists = squeeze(mapslices(norm, points - pred_points, 1), 1)
    println(" Done.")

    print("Computing quantities...")
    x = count(dists .< y)/length(dists)
    q = Base.quantile(dists, p)
    println(" Done.")

    println("(P(<y), $p-%tile) = ", (x, q))

    (x, q)
end
