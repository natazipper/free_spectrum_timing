# free_spectrum_timing
Creating free spectrum from par and tim files. Prepare the input datafile for ceffyl
Usage
-to create free spectrum
python free_spec.py epta_sim/ 30 1000000 report_sim/

epta_sim is the input directory with par and tim files
30 is the number of frequency components
1000000 is the number of iterations in the mcmc
report_sim is the output dierctory for the mcmc

-to create an input file for ceffyl
python ceffyl_file.py epta_sim/ 30 report_sim/ spec_ceffl/

epta_sim is the input directory with par and tim files
30 is the number of frequency components
spec_ceffyl is the output directory for the ceffyl input files

-produce frequency resolved optimal statistics
python single_bin_snr.py epta_sim/ report_sim/ 30

epta_sim is the input folder with par and tim files
report_sim is the folder with the chain of the free spectrum
30 is the number of freuquency components (should coincide with the the number of components of the free spectrum)
