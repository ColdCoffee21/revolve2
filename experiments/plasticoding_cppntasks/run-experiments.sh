#!/bin/bash

study="plasticoding_cppntasks"
mainpath="/storage/karine"
screen -d -m -S run_loop -L -Logfile ${mainpath}/${study}/setuploop.log ./experiments/${study}/setup-experiments.sh