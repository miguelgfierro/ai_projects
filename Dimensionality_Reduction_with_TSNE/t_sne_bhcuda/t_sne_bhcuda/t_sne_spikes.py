
from os.path import dirname, join
import h5py as h5
import numpy as np
import t_sne_bhcuda.bhtsne_cuda as TSNE
import time


def t_sne_spikes(kwx_file_path, hdf5_dir_to_pca=r'channel_groups/1/features_masks', mask_data=False,
                 path_to_save_tmp_data=None, indices_of_spikes_to_tsne=None, use_scikit=False, perplexity=50.0,
                 theta=0.5, iterations=1000, seed=0, gpu_mem=0.8, no_dims=2, eta=200, early_exaggeration=4.0,
                 randseed=-1, verbose=2):
    """
    Uses the PCA (masked or not) results of the spikedetection that the phy module does to embed the 3 X N_channels
    dimensional spikes into no_dims (usually 2) dimensions for visualization and faster manual sorting purposes.
    The embeding is done using the t-sne algorithm. The GUI of the phy module reads the result of the t-sne and plots
    them superimposing and sorting information.

    For embeding spikes using other features (like T_timepoints x N_channels for eaxample) then use directly the
    bhtsne_cuda.t_sne() function.

    Parameters
    ----------
    kwx_file_path -- The path where the .kwx file is resulting from phy's spikedetect (the file that has the PCA and
     mask results for each spike)
    hdf5_dir_to_pca -- The path in the kwx (hdf5) file that the pca and mask matrices are saved in
    mask_data -- Use the masking that spikedetect provides on the PCA results or not
    path_to_save_tmp_data -- If it is not set, the t-sne intermediate files (see bhtsne_cuda.py) will be saved in
    the kwx_file_path.
    indices_of_spikes_to_tsne -- Choose a subgroup of spikes to t-sne from the ones spikedetect found
    use_scikit -- If True then use the sklearn t-sne implementation (Python, no GPU).
    perplexity -- Defines the amount of samples whose distances are comparred to every sample (check sklearn and the
    van der Maatens paper)
    theta -- If > 0 then the algorithm run the burnes hat aproximation (with angle = theta). If = 0 then it runs the
     exact version. Values smaller than 0.5 do not add to much error.
    iterations -- The number of itterations (usually around 1000 should suffice)
    gpu_mem -- If > 0 (and <= 1) then the t_sne_bhcuda.exe will run the eucledian distances calculations on the GPU
    (if possible) and will use (gpu_mem * 100) per cent of the available gpu memory to temporarily store results. If
    == 0 then the t_sne_bhcuda.exe will run only on the CPU. It has no affect if use_scikit = True
    no_dims -- Number of dimensions of the t-sne embedding
    eta -- The learning rate
    early_exaggeration -- The amount by which the samples are initially pushed apart
    randseed -- Set the random seed for the initiallization of the samples on the no_dims plane.
    verbose -- Define verbosity. 0 = No output, 1 = Basic output, 2 = Full output, 3 = Also save t-sne results in
    interim files after every iteration. Option 3 is used to save all steps of t-sne to explore the way the algorithm
    seperates the data (good for movies).

    Returns
    -------
    A N_examples X no_dims array of the embeded spikes. It also saves the same array in a .npy file in the kwx_file_path
    that can be read by the GUI of the phy module
    """

    h5file = h5.File(kwx_file_path, mode='r')
    pca_and_masks = np.array(list(h5file[hdf5_dir_to_pca]))
    h5file.close()
    masks = np.array(pca_and_masks[:, :, 1])
    pca_features = np.array(pca_and_masks[:, :, 0])
    masked_pca_features = pca_features
    if mask_data:
        masked_pca_features = pca_features * masks

    if indices_of_spikes_to_tsne is None:
        num_of_spikes = np.size(masked_pca_features, 0)
        indices_of_spikes_to_tsne = range(num_of_spikes)
    data_for_tsne = masked_pca_features[indices_of_spikes_to_tsne, :]

    if not path_to_save_tmp_data:
        if verbose:
            print('The C++ t-sne executable will save data (data.dat and results.data) in \n{}\n'
                  'You might want to change this behaviour by supplying a path_to_save_tmp_data.\n'.
                  format(dirname(kwx_file_path)))
        path_to_save_tmp_data = dirname(kwx_file_path)

    del pca_and_masks
    del masks
    del pca_features
    del masked_pca_features

    t0 = time.time()
    t_tsne = TSNE.t_sne(data_for_tsne, use_scikit=use_scikit,files_dir=path_to_save_tmp_data,
                        no_dims=2, perplexity=perplexity, eta=eta, theta=theta, iterations=iterations, seed=seed,
                        early_exaggeration=early_exaggeration, gpu_mem=gpu_mem, randseed=randseed, verbose=verbose)
    t_tsne = np.transpose(t_tsne)
    t1 = time.time()
    if verbose > 1:
        print("CUDA t-sne took {} seconds, ({} minutes)".format(t1-t0, (t1-t0)/60))

    np.save(join(dirname(kwx_file_path), 't_sne_results.npy'), t_tsne)

    return t_tsne
