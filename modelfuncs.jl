export @modelfunc

using Plots
using StatsBase: mode
import JLD

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

@modelfunc cachedists
function cachedists(model, key_file, events, points)
    key, file = key_file

    print("Computing predictions...")
    preds = predict(model, events)
    println(" Done.")

    events = permutedims(events, [3, 1, 2])
    points = permutedims(points, [2, 1])
    preds = permutedims(preds, [2, 1])

    print("Computing distances...")
    dists = mapslices(norm, preds - points, 2) |> @λ squeeze(_, 2)
    println(" Done.")

    print("Saving distances to ", file, " as ", key, "...")
    JLD.save(file, key, dists)
    println(" Done.")
end

@modelfunc pointhist
pointhist(model, args...; model_name="", kws...) =
    pointhist([model], args...; model_name=[model_name], kws...)
pointhist(model::AbstractVector, args...; kws...) =
    MethodError(pointhist, (model, args...)) |> throw
function pointhist(models::AbstractVector, events, points, y, p; model_name=[], color=:auto)
    print("Computing predictions...")
    preds = map(x -> predict(x, events), models)
    println(" Done.")
  
    print("Computing distances...")
    dists = map(x -> squeeze(mapslices(norm, x - points, 1), 1), preds)
    println(" Done.")

    print("Generating histogram...")

    color isa AbstractVector && (color = @reshape color[_, :])
    linhist1 = stephist(dists, label=model_name, color=color)
    linhist2 = stephist(dists, legend=false, xlims=(0, 4), color=color)
    loghist1 = stephist(dists, legend=false, yaxis=(:log10, (1, Inf)), color=color,
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
