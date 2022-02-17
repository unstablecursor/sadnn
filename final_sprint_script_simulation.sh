#!/bin/bash
for exc in $(seq 0.5 .1 1.6)
do
    for inh in $(seq 5.0 1.0 16.0)
    do
        timeout -k 1m 10m python3 final_best_script.py $exc $inh | tee -a carrada_spike_log.txt
        pkill -f final_best_script.py -9
        pkill -f pyth -9
        sleep 5
    done
done
