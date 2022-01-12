#!/bin/bash

for it_sigma_exc in 0.90
    do
    for it_sigma_inh in 11.0
        do
        for it_intensity_exc in 0.7 0.8 0.9 1.0 1.1 1.2 1.3
            do
            for it_intensity_inh in -7.5 -8.5 -9.5 -10.5 -11.5 -12.5 -13.5
                do
                timeout -k 1m 15m python3 simulation_working_spiking.py $it_sigma_exc $it_sigma_inh $it_intensity_exc $it_intensity_inh | tee -a evaluate_sim_spike_logs.txt
                pkill -f simulation_working_spiking.py -9		
                pkill -f pyth -9
                sleep 5
                done
            done
        done
    done

