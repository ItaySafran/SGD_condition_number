SGD_condition_number
====================
Code used in the paper [Random Shuffling Beats SGD Only After Many Epochs on Ill-Conditioned Problems]().

This repository contains the following two files:
-----------------------------------------
* 'with- vs without-replacement SGD.py', a Python 3.6 file which is used to run the experiments in Sec. 5 in the paper. The file does not require any special packages or libraries to run.
* An additional ZIP file 'experiment_output.zip' containing 150 files with an extension ".p". These files are the output generated when running the code, which allows running the code, pausing, and then resuming its execution. The file names begin with either "ss" (for output obtained from running on the construction in Eq. (1) in the paper) or "rr" (for output obtained from running on the construction in Eq. (2) in the paper) and contain a certain value of the parameter k in the file name, representing the number of epochs used in the result in the file (as well as the step size that was used and depends on k - see Sec. 5 in the paper for further explanation). In each such file, the average loss of 100 SGD instanstiations ran for k epochs is stored. More specifically, each such file contains four values, as elaborated below:

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;1. result[0] - The sum of the losses attained using with-replacement SGD.

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;2. result[1] - The sum of the losses attained using without-replacement SGD with a single shuffling.

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;3. result[2] - The sum of the losses attained using without-replacement SGD with random reshuffling.


Running the code:
-----------------
Does not require installing any special packages, just requires Python 3.6 and numpy installed.


Output files:
-------------
Once the code is ran, the 150 files in the ZIP file 'experiment_output.zip' are generated (one can change the number of instantiations in the code to produce such an output more quickly). Then the graph appearing in Figure 1 in the paper is generated.
