#!/bin/bash
#set -e
#set -x


study="plasticoding_cppntasks"

# DO NOT use underline ( _ ) in the experiments names
# delimiter is space, example:
#experiments=("exp1" "epx2")
# exps order is the same for all params
# exps names should not be fully contained in each other

experiments=("nonplasticforthright" "plasticforthright" "nonplasticbackforth" "plasticbackforth" "fullplasticforthright" "fullplasticbackforth" "onlyforth")
population_size=(200 200 200 200 200 200 100)
offspring_size=(200 200 200 200 200 200 100)
num_generations="100"


fitness_measure=("forthright_dominated" "forthright_dominated" "backforth_dominated" "backforth_dominated" "forthright_dominated" "backforth_dominated" "speed_y" )
seasons_conditions=("1.0_1.0_0_0_0#1.0_1.0_0_0_1" "1.0_1.0_0_0_0#1.0_1.0_0_0_1" "1.0_1.0_0_0_0#1.0_1.0_0_0_1" "1.0_1.0_0_0_0#1.0_1.0_0_0_1"  "1.0_1.0_0_0_0#1.0_1.0_0_0_1" "1.0_1.0_0_0_0#1.0_1.0_0_0_1" "1.0_1.0_0_0_0")
plastic_body=(0 0 0 0 1 1 0)
plastic_brain=(0 1 0 1 1 1 0)

simulation_time=30
runs=20

num_terminals=2
mainpath="/storage/karine"

mkdir ${mainpath}/${study}
mkdir ${mainpath}/${study}/analysis

possible_screens=()

# defines possible ports for screens
for t in $(seq 1 $((${num_terminals}))); do
    possible_screens+=($t)
done


while true
	do

    printf "\n  >>>> loop ... \n"

    # discover free terminals

    active_screens=()
    free_screens=()
    active_experiments=()


    declare -a arr="$(screen -list)"


    for obj in ${arr[@]}; do

        if [[ "$obj" == *"screen_"* ]]; then
          printf "\n screen ${obj} is on\n"
          screen="$(cut -d'_' -f2 <<<"$obj")"
          active_experiments+=("$(cut -d'_' -f3 -<<<"$obj")_$(cut -d'_' -f4 -<<<"$obj")")
          active_screens+=($screen)
        fi
    done

   for possible_screen in "${possible_screens[@]}"; do
       if [[ ! " ${active_screens[@]} " =~ " ${possible_screen} " ]]; then
           free_screens+=($possible_screen)
     fi
      done


    # discover unfinished experiments

    to_do=()
    unfinished=()
    for i in $(seq $runs)
    do
        run=$(($i))

        for experiment in "${experiments[@]}"
        do

         printf  "\n${experiment}_${run} \n"
         file="${mainpath}/${study}/${experiment}_${run}.log";

         #check experiments status
         if [[ -f "$file" ]]; then

              lastgen=$(grep -c "Finished generation" $file);
              echo "latest finished gen ${lastgen}";

             if [ "$lastgen" -lt "$num_generations" ]; then
                 unfinished+=("${experiment}_${run}")

                # only if not already running
                if [[ ! " ${active_experiments[@]} " =~ " ${experiment}_${run} " ]]; then
                   to_do+=("${experiment}_${run}")
                fi
             fi
         else
             # not started yet
             echo " None";
               unfinished+=("${experiment}_${run}")
               # only if not already running
                if [[ ! " ${active_experiments[@]} " =~ " ${experiment}_${run} " ]]; then
                   to_do+=("${experiment}_${run}")
                fi
         fi

        done
    done


    # spawns N experiments (N is according to free screens)

    max_fs=${#free_screens[@]}
    to_do_now=("${to_do[@]:0:$max_fs}")

    p=0
    for to_d in "${to_do_now[@]}"; do

        exp=$(cut -d'_' -f1 <<<"${to_d}")
        run=$(cut -d'_' -f2 <<<"${to_d}")
        idx=$( echo ${experiments[@]/${exp}//} | cut -d/ -f1 | wc -w | tr -d ' ' )

        # nice -n19 python3  experiments/${study}/optimize.py
        screen -d -m -S screen_${free_screens[$p]}_${to_d} -L -Logfile /storage/karine/${study}/${exp}_${run}".log" python3  experiments/${study}/optimize.py \
               --experiment_name ${exp}  --study=${study}  --seasons_conditions ${seasons_conditions[$idx]} --run ${run} --fitness_measure ${fitness_measure[$idx]} \
               --plastic_body ${plastic_body[$idx]} --plastic_brain ${plastic_brain[$idx]} --num_generations ${num_generations} \
               --offspring_size ${offspring_size[$idx]} --population_size ${population_size[$idx]} --simulation_time ${simulation_time};

        printf "\n >> (re)starting screen_${free_screens[$p]}_${to_d} \n\n"
        p=$((${p}+1))

    done

   # if all experiments are finished, makes video
   if [ -z "$unfinished" ]; then
       file="${mainpath}/${study}/analysis/video_bests.mpg";

     if [ -f "$file" ]; then
        printf ""
     else
         printf " \n making video..."
         screen -d -m -S videos ffmpeg -f x11grab -r 25 -i :1 -qscale 0 $file;
         python3 experiments/${study}/watch_robots.py;
         killall screen;
         printf " \n finished video!"
      fi
    fi


    sleep 600;

done
