#!/bin/bash
# run this script from the root (revolve folder): ./experiments/bodybrain_evo/run-experiments.sh PARAMSFILE

DIR="$(dirname "${BASH_SOURCE[0]}")"
study="$(basename $DIR)"

if [ $# -eq 0 ]
  then
     params_file=paramsdefault
  else
    params_file=$1
fi

source $DIR/$params_file.sh

mkdir ${outputs_path}/${study};
screen -d -m -S run_loop -L -Logfile ${outputs_path}/${study}/setuploop.log $DIR/setup-experiments.sh ${params_file};

### CHEATS: ###

# to check all running exps screens: screen -list
# to stop all running exps: killall screen
# to check a screen: screen -r naaameee
# screen -ls  | egrep "^\s*[0-9]+.screen_" | awk -F "." '{print $1}' |  xargs kill