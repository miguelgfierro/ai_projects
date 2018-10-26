# T-SNE with CUDA and Barnes Hut (with Python wrapper) with a spike sorting application

## T-sne code

This is an extension of the C++ t-sne with barnes-hut algorithm written by Laurens van der Maaten. The original code can be found here: [lvdmaaten/bhtsne](https://github.com/lvdmaaten/bhtsne/). There are five main differences.


1. The cuda code targets not the actual t-sne code but the part that generates the perplexities. More specifically it implements a fast cuda algorithm for calculating euclidean distances. That way the most time consuming step in the barnes-hut algorithm (calculating the euclidean distances of sample pairs) becomes much fater.
The code for the euclidean distance calculation in cuda was taken from here:
*Chang, Darjen, Nathaniel A. Jones, Dazhuo Li, and Ming Ouyang. “Compute Pairwise Euclidean Distances of Data Points with GPUs.” In Proceedings of the IASTED International Symposium Computational Biology and Bioinformatics. Orlando, Florida, USA, 2008.*
The code for the dev_array was taken from Valerio Restocchi's blog, here: [dev_array: A Useful Array Class for CUDA](https://www.quantstart.com/articles/dev_array_A_Useful_Array_Class_for_CUDA).
Currenlty the cuda code works only with the bh part of the original code (if the theta parameter is set to larger than 0). The exact part of the code (theta = 0) is left the same (requires BLAS or a windows equivalent to calculate the euclidean distances). In the future the cuda euclidean distances calculation will be implemented in the exact part of the code to allow even faster calculation than BLAS.
The current cuda implementation checks the amount of available gpu memory and uses a user defined percentage of it (set by default to 80% of the available gpu memory). If you set the gpu_mem parameter to 0 then the code runs on the cpu (as per the original). A value larger than 0 and smaller than 1 tells the program to use that percentage of the available gpu memory. If the gpu memory required to store all distance pairs (4\*N\*N bytes where N is the number of samples) is larger than what the gpu can offer then the algorithm iterates the saving of the distances in chunks that can be temporarily held in gpu memory. Of course if the available RAM is smaller than 4\*N\*N then the program crashes (maybe a mmap implementation in the future wouldn't add too much time in the reading of the distances from hard disk).
The cuda was written using CUDA 7.5 (January 2016) but should work with anything over 5.0 (untested claim). The code generation is set to compute_35,sm_35 but compute_20,sm_20 might still work (again untested claim).

2. The second change is that the code now is a Visual Studio (2013) project using the nvcc compiler (through the Nsight Visual Studio Edition 4.0). Maybe there will be a make file in the future for cross platform compilation but for now if you are using Linux or Mac you will have to make your own project. There aren't any windows specific libraries and I have used int_least64_t instead of long long to make matters a bit easier. The code has also been compiled on a Linux machine by importing it into an Eclipse Nsight project and worked without any modifications. So setting up your own project should be easy. Also you might want to have a look at the original Maatens t-sne code on ideas on how to get started in \*nix* systems. 

3. For the cases where the amount of samples is too large (i.e the 4\*N\*N distance matrix bytes do not fit in RAM) the algorithm implements a way to use part of the samples as a template for the rest. If the seed parameter is set to any number X > 0 and < Nsamples then only the first X samples will go through the t-sne code. For the remaining N - X samples the following procedure is followed. Each one of the remaining samples has its distance to all the X samples calculated (on the GPU). The 5 closest samples are selected and their t-sne coordinates averaged. The new sample coordinates on the t-sne space are set to these averages. This procedure assumes that the N - X samples that were not part of the original t-sne do not form part of any extra clusters. In order for this assumption to be satisfied a good precaution is to randomize the sample order before the data are given to the algorithm. 
 
4. The fourth difference is a small re-writing of the python wrapper (originaly developed by Pontus Stenetorp). Now you can use it to generate the data.dat file or read the results.dat file without actually running the t-sne code. The data.dat file also carries a header that passes the required parameters into the t_sne_gpu executable. It also allows the user to call the sklearn implementation of t-sne (no cuda, no C++ = quite more slow) in case something has gone wrong and the C++ executable does not run.

5. The final difference is that now the C++ code allows the output (saved as interim data files) of the t-sne process at every itteration. This can be turned on by seting the verbose parameter to bigger than 2.

## Spike-sorting application
On top of the extensions in the C++ there is python code to use t-sne to do clustering of neural spikes and to also run a gui that allows manual clustering based on the t-sne results.

Specifically the spike sorting application (t_sne_spikes.py, tsne_cluster.py and the helper functions in spike_heatmap.py) is meants to extend the capabilities of the [phy module](http://phy.readthedocs.org/en/latest/) by adding t-sne visualization and t-sne based manual sorting capabilities. It assumes that phy has been used to detect the spikes in a dataset and the t_sne_spikes function uses the masked PCA components phy detect generates to t-sne the spikes. At a second step the gui_manual_cluster function in the tsne_cluster.py script can be called with the t-sne results to allow for manual sorting of the spikes using the 2D embedding of the spikes. The GUI allows the user to select groups of spikes and deside if this is a unit by checking:

1. the time plot of all channels averaged over the selected spikes

2. the autocorrelogram of the selected spikes

3. the heatmap of the peak to peak amplitude of all channels averaged over the selected spikes (requires the use of the .prb probe file that has been used by phy to detect the spikes).

The GUI saves a pandas dataframe (as a pickle file) with the name given to every cluster, the number of spikes in it and the indices of the spikes (in the t-sne dataset).
The GUI is generated using the bokeh library and requires running a bokeh server (once bokeh in installed do 'bokeh serve' in a command prompt).
The python code is kept at the level of a number of functions (no object orientation) to allow users to 'easily' add functionality to the GUI.  

## Notes for use
### Installation
 This is a conda package so you can install it using conda install -c georgedimitriadis t_sne_bhcuda (after you have done conda install anaconda-client to install access to the anaconda cloud where the code is hosted). This will add the t_sne_bhcuda executable into the Scripts folder (for Windows) or the bin folder (for Linux) of the python environment that you installed the package in. The t_sne() function in the bhtsne_cuda script of the module will call this executable.

 The whole code base (including the C++ and CUDA source code) will be downloaded into your Conda_folder\conda-bld\work.

 **Important note for Windows users:** If you have not used cuda before, then you need to be aware that windows by default will stop and restart the nvidia driver if it thinks that the gpu is stuck. That by default will happen if the gpu does anything that takes longer than 2 seconds. The current code will not work under these conditions with sample sizes over a certain number. If the code requires more than 2 seconds to calculate the distances then windows will restart the driver and the program will fail (you will get a notification of this at the bottom of your screen). In order to get windows off your back do what he says: [Nvidia Display Device Driver Stopped Responding And Has Recovered Successfully (FIX)](https://www.youtube.com/watch?v=QQJ9T0oY-Jk). Also have a look here for MSDN info on the relative registry values [TDR Registry Keys](https://msdn.microsoft.com/en-us/library/windows/hardware/ff569918%28v=vs.85%29.aspx).

### Use
Read the example in the bhtsne_cuda script on how to call t_sne. A very extensive documentation for the t-sne specific parameters can also be found in the sklearn.manifold.TSNE package.

Not t-sne related parameters are the gpu_mem, the seed and verbose. The gpu_mem needs to be set between 0 and 1 and will control the percentage of GPU memory (from the amount available) each GPU cycle will use. If the number of distances calculated needs more than available*gpu_mem bytes then the computation is split in cycles that fit into that memory. 0 means no GPU is used (the code defaults to the original barnes hat algorithm on the CPU).
The seed is the number of samples from the total data samples that will actually go through t-sne. It needs to be between 0 and Nsamples. 0 (or Nsamples) means just run all the samples normally through t-sne. Any other number X means the remaining N-X samples are placed on the t-sne space through a process described in point 3 above.
Verbose will both define the amount of feedback t-sne will produce and (if set > 2) will save all intermediate iterations of t-sne in seperate files (good for making movies of the t-sne process).

If you want to use t-sne on spikes as detected by the spikedetect part of the phy module then check out the t_sne_spikes script. The result of this operation is a 2 x Nspikes array that will be saved in the same directory that the .kwik file you provided to the function is in. The phy module will be able to detect this array (saved in .npy format) and display in its GUI the results of the t-sne operation superimposing clustering information if it exists. Or you can use the gui_manual_cluster function to do a fully manual spike sorting.
