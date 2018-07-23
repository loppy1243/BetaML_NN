export @λ, @reshape, fcat, xrel, yrel

import Plots

using Loppy.Util

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
