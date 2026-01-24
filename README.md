# free_spectrum_timing
1. Generate tim file with injected Omega_gw, which is a power law (with given amplitude and spectral index nt):
   ```
   python inject_omega.py --datadir epta_sim/ --comp 30 --iter_num 1e5 --iter_real 1 --datadir_out output_3/ --nt 3.4 --amp 1e-10
   ```
All the output will be saved in datadir_out folder

3. Generate tim files with injected Omega_gw with fully customised shape.
First we need to generate a file with a customised spectra:
   ```
   python generate_spec.py
   ```

Secondly, we need to run a code with the following options:
   ```
   python inject_omega.py --datadir epta_sim/ --comp 30 --iter_num 1e5 --iter_real 1 --datadir_out output_4/ --cust_spec cust_spec.txt
   ```
   
Examples: 

output_3/ --nt 3.4 --amp 1e-10

output_2/ --nt 1.4 --amp 1e-9

output_6/ --nt 3.4 --amp 5e-9

output_4/
amp = 1e-9 #amplitude at 1 year
nt = 2.4
amp_g = 2*amp #in comparison to the amplitude at 1 year
sigma_g = 0.2
centr_fr = 1e-8

output_5/
amp = 1e-9 #amplitude at 1 year
nt = 2.4
amp_g = 10*amp #in comparison to the amplitude at 1 year
sigma_g = 0.2
centr_fr = 5e-8

