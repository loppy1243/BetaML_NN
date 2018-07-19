export @modelfunc

import JLD2

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

#macro modelfunc(funcname)
#    fname = esc(funcname)
#
#    quote
#        $fname(modelfile::AbstractString, events, cells, args...; kws...) =
#            $fname(BetaML_NN.load(modelfile), events, cells, args...; kws...)
#    end
#end

function pointhist(model, args...; key=nothing, model_name="", kws...)
    key === nothing && error("Must specify keys to load distances")
    pointhist([model], args...; key=[key], model_name=[model_name], kws...)
end
pointhist(model::AbstractVector, args...; kws...) =
    MethodError(pointhist, (model, args...)) |> throw
function pointhist(models::AbstractVector, y, p;
                   key=String[], model_name=String[], color=:auto, xlims=(0.0, 4.0),
                   size=(500, 500))
    isempty(key) && error("Must specify keys to load distances")

    print("Loading distances...")
    dists = JLD2.jldopen("data/dists.jld2", "r") do io
        dists = Array{Float64}(length(io[first(key)]), length(key))
        for (i, k) in enumerate(key)
            dists[:, i] .= io[k]
        end
        dists
    end
    println(" Done.")

    print("Generating histogram...")

    color isa AbstractVector && (color = @reshape color[_, :])
    model_name = @reshape model_name[_, :]

    linhist1 = histogram(dists, title="Full", label=model_name, color=color)
    linhist2 = histogram(dists, title="Closeup", legend=false, xlims=xlims, color=color)
    loghist1 = histogram(dists, title="Log", legend=false, yaxis=(:log10, (1, Inf)),
                        color=color, xlabel="Dist. of pred. from true (mm)")
    normed = histogram(dists, normed=true, title="Normed", legend=false, xlims=xlims,
                      color=color, xlabel="Dist. of pred. from true (mm)")

    hist = plot(layout=(1, 2), plot(layout=(2, 1), linhist1, loghist1),
                               plot(layout=(2, 1), linhist2, normed),
                               size=size)
    println(" Done.")

    print("Computing statistics...") 
    ms = mapslices(minimum, dists, 1)
    Ms =  mapslices(maximum, dists, 1)
    mes = mapslices(mean, dists, 1)
    mos = mapslices(mode, dists, 1)
    sts = map(i-> std(dists[:, i], mean=mes[i]), indices(dists, 2))
    xs = mapslices(dists, 1) do ds
        count(ds .< y)/length(ds)
    end
    qs = mapslices(dists, 1) do ds
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
