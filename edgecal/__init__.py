"""Edge calibration: the three anchors -- R_min, R_p99, R_max -- that `pricing/edge` uses to turn a
risk number into a required edge.

Only the loader is published. The calibration itself (the Greek sweep over a year of real boards,
the fit, and its validation gate) is private research, as is the fitted artifact it reads.
"""
