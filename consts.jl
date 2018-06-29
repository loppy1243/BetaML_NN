# EVENTS from "readdata.jl"

const DATAFILE = "I:\\projects\\temp\\liddick\\BetaScint2DEnergy.txt"
const TRAIN_RANGE = 1 : div(EVENTS, 3)
const VAL_RANGE = div(EVENTS, 3)+1 : 2*div(EVENTS, 3)
const TEST_RANGE = 2*div(EVENTS, 3)+1 : EVENTS
const BATCHES = 1000
const BATCHSIZE = div(EVENTS, BATCHES)
const EPOCHS = BATCHES
const MAX_E = 3060
