#!/bin/bash

# delimiter is comma, example:
#experiments="exp1,epx2"
# exps order is the same for all params

experiments="nonplasticforthright,plasticforthright,nonplasticbackforth,plasticbackforth,fullplasticforthright,fullplasticbackforth,onlyforth"
# these params are the same for all exps
# gens for boxplots and snapshots
generations=(100)
#gen for lineplots
final_gen=100
runs=20
mainpath="karine"
study="plasticoding_cppntasks"

#python experiments/${study}/snapshots_bests.py $study $experiments $runs $generations $mainpath;
#python experiments/${study}/bests_snap_2d.py $study $experiments $runs $generations $mainpath;
#python experiments/${study}/consolidate.py $study $experiments $runs $final_gen $mainpath;

experiments="nonplasticforthright,plasticforthright,fullplasticforthright"
comparison='forthright'
python experiments/${study}/plot_seasonal.py $study $experiments $runs $generations $comparison $mainpath;
#python experiments/${study}/plot_trajectory.py $study $experiments $runs $final_gen $comparison $mainpath;

experiments="nonplasticbackforth,plasticbackforth,fullplasticbackforth"
comparison='backforth'
python experiments/${study}/plot_seasonal.py $study $experiments $runs $generations $comparison $mainpath;
#python experiments/${study}/plot_trajectory.py $study $experiments $runs $final_gen $comparison $mainpath;

# METADATA:

# backforth means that in cond 1 speed_y should be max and in cond 2 speed_y should be min
# back and forth are visually left and right respectively (in practice, first it goes forth and then back)

# forthright means that in cond 1 speed_y should be max and in cond 2 speed_x should be max
# forth and right are visually right and down respectively