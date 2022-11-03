    #!/bin/bash


### PARAMS INI ###

# this should be the path for the output files (choose YOUR OWN dir!)
outputs_path="/storage/karine"

# DO NOT use underline ( _ ) in the experiments names
# delimiter is space, example:
#experiments=("exp1" "epx2")
# exps order is the same for all params
# exps names should not be fully contained in each other

study="default_study"
experiments=("defaultexperiment")
seasons_conditions=("1.0_1.0_0_0_0")

runs=2

# use num_generations=100 for interesting results, and num_generations=3 for quick test
num_generations="3"

# use population_size=100 for interesting results, and population_size=11 for quick test
population_size="11"

# use offspring_size=100 for interesting results, and offspring_size=11 for quick test
offspring_size="11"

# use delay_setup_script=1800 for interesting (longer) experiments
delay_setup_script=60

# recommended not more than two in the rippers
num_terminals=2

# gens for boxplots and snapshots (by default the last gen)
#generations="1,$num_generations"
generations="$num_generations"

# max gen to filter lineplots  (by default the last gen)
final_gen="$num_generations"


### PARAMS END ###