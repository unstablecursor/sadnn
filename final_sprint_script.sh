#!/bin/bash
cat carrada_sets.txt | while read line 
    do
    timeout -k 1m 15m python3 final_best_script.py $line | tee -a carrada_spike_log.txt
    pkill -f final_best_script.py -9		
    pkill -f pyth -9
    sleep 5
    done
