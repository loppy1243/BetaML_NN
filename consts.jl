export DATAFILE, TRAIN_RANGE, VALID_RANGE, TEST_RANGE, BATCHSIZE, MAX_E

using BetaML_Data: EVENTS

const DATAFILE = "I:\\projects\\temp\\liddick\\BetaScint2DEnergy.txt"
const TRAIN_RANGE = 1 : div(EVENTS, 3)
const VALID_RANGE = div(EVENTS, 3)+1 : 2*div(EVENTS, 3)
const TEST_RANGE = 2*div(EVENTS, 3)+1 : EVENTS
const BATCHSIZE = 1000
const MAX_E = 3060
