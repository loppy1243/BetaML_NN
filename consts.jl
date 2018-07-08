export DATAFILE, TRAIN_RANGE, VALID_RANGE, TEST_RANGE, BATCHSIZE, MAX_E

using BetaML_Data: EVENTS

isfile("consts.local.jl") && include("consts.local.jl")

@try_defconst DATAFILE = "BetaScint2DEnergy.txt"
@try_defconst TRAIN_RANGE = 1 : div(EVENTS, 3)
@try_defconst VALID_RANGE = div(EVENTS, 3)+1 : 2*div(EVENTS, 3)
@try_defconst TEST_RANGE = 2*div(EVENTS, 3)+1 : EVENTS
@try_defconst BATCHSIZE = 1000
