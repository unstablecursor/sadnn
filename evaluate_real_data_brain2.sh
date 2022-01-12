#!/bin/bash

for it_sigma_exc in 1.0
    do
    for it_sigma_inh in 5.0
        do
        for it_intensity_exc in 1000.0 1400.0
            do
            for it_intensity_inh in -1000.0 -1200.0 -800.0
                do
                for lower_volt_thresh in -1.0
                    do
                    timeout -k 1m 15m python3 EvalSpiking_test_working.py $it_sigma_exc $it_sigma_inh $it_intensity_exc $it_intensity_inh $lower_volt_thresh | tee -a log.txt
                    pkill -f EvalSpiking.py -9		
                    pkill -f pyth -9
                    sleep 5
                    done
                done
            done
        done
    done

