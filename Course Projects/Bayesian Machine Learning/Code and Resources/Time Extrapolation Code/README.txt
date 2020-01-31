In both folders you will find code written by Max Jenquin, C.E. Rasmussen (as part of the GPML codebase for Gaussian Process Regression), Maziar Raissi, and A.G. Wilson. Code by other authors is generally available online, and citations of relevant papers have been made in the project report this folder accompanies.

To reproduce results, navigate to the "Examples" folder in either "RBF Integration" or "SM Integration". Run "XX_integrator_xx.m", where XX refers to the equation being integrated and xx refers to the kernel involved. Results will be automatically plotted, but a log of the simulation will also be saved as a .mat file in the corresponding "Results" folder. You may optionally run only a clean data simulation or a noisy data simulation.

Playing with parameters in the SM case is not advised, very few other settings converge to a reasonable result with any probability (always dependent on rng in optimizer though). However, if you find another parameter regime that seems to work well, I'll be very interested to hear about it, email me at mrj89@cornell.edu

A list of included results:
RBF Integration/Results/
	rbfKDVresults.mat		complete clean and noisy simulations
	rbfKSresults.mat		complete clean and noisy simulations
	rbfNLSresults.mat		complete clean and noisy simulations (appendix content)
SM Integration/Results/
	smKDVresults.mat		complete clean and noisy simulations
	smKSresults.mat			complete clean and noisy simulations
	smKSresults_midwaysample.mat 	clean simulation for a sample point within the chaotic 							regime, a qualitatively different regression task

- Max Jenquin, Cornell University: Center for Applied Mathematics, Dec. 2018