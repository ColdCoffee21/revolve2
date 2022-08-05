#!/bin/bash

study="default_study"
# make sure to provide experiments names in alphabetic order
# arrays use comas in this case
#experiments=("exp1,epx2")
experiments=("defaultexperiment")
mainpath="karine"
runs=10
generations=(100)
final_gen=100

python experiments/${study}/snapshots_bests.py $study $experiments $runs $generations $mainpath;
python experiments/${study}/bests_snap_2d.py $study $experiments $runs $generations $mainpath;
python experiments/${study}/consolidate.py $study $experiments $runs $final_gen $mainpath;
python experiments/${study}/plot_static.py $study $experiments $runs $generations $mainpath;
