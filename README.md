# free_spectrum_timing
1. Generate tim file with injected Omega_gw, which is a power law (with given amplitude and spectral index nt):

python inject_omega.py --datadir epta_sim/ --comp 30 --iter_num 1e5 --iter_real 1 --datadir_out output_3/ --nt 3.4 --amp 1e-10

All the output will be saved in datadir_out folder

2. Generate tim files with injected Omega_gw with fully customised shape:

python inject_omega.py --datadir epta_sim/ --comp 30 --iter_num 1e5 --iter_real 1 --datadir_out output_4/ --cust_spec cust_spec.txt

Examples: 

output_3/ --nt 3.4 --amp 1e-10

output_2/ --nt 1.4 --amp 1e-9
