#!/bin/bash

for n in {1..20..1};
do
   python inject_omega.py epta_sim_1/ 30 50000 report_sim1/ $n
done
