#!/usr/bin/env bash

mpirun -n 2 python3 co-sim-arbor.py --final-time 20000 --time-step 0.01 --cell-count 1000 --weight 0.5 --k-bath-bad 17.0 --k-bath-ok 9.5 --pathological-fraction 1.0 --beta 0.1 --proxy-region 72
