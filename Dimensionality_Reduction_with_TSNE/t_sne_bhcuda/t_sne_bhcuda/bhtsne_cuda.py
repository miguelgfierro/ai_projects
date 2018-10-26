#!/usr/bin/env python

"""
A simple Python wrapper for the t_sne_bhcuda binary.

Note: The script does some minimal sanity checking of the input, but don't
    expect it to cover all cases. After all, it is a just a wrapper.

This code is a small extension to the original python wrapper by Pontus Stenetorp
which passes to the t_sne_bhcuda executable the amount of gpu
memory to be used. It also splits the read, write and execute parts into separate
functions that can be used independently.

It also acts as a thin wrapper to the scikit learn t-sne implementation
(which can be called instead of the t_sne_bhcuda executable).

The data into the t_sne function (or the save_data_for_tsne function) is a samples x features array.

Example

import t_sne_bhcuda.bhtsne_cuda as tsne_bhcuda
import matplotlib.pyplot as plt

perplexity = 50.0
theta = 0.5
learning_rate = 200.0
iterations = 2000
gpu_mem = 0.8
t_sne_result = tsne_bhcuda.t_sne(samples=data_for_tsne, files_dir=r'C:\temp\tsne_results',
                        no_dims=2, perplexity=perplexity, eta=learning_rate, theta=theta,
                        iterations=iterations, gpu_mem=gpu_mem, randseed=-1, verbose=2)
t_sne_result = np.transpose(t_sne_result)

fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(t_sne_result[0], t_sne_result[1])

Original Author:    Pontus Stenetorp
Author:             George Dimitriadis    <george dimitriadis uk>
Version:            0.2.0
"""

from os.path import abspath, pardir, dirname, isfile, isdir, join as path_join
from struct import calcsize, pack, unpack
from subprocess import Popen, PIPE
from platform import system
import sys
import os
import numpy as np


# Default hyper-parameter values
DEFAULT_NO_DIMS = 2
DEFAULT_PERPLEXITY = 50.0
DEFAULT_EARLY_EXAGGERATION = 4.0
DEFAULT_THETA = 0.5
DEFAULT_RANDOM_SEED = -1
DEFAULT_ETA = 200
DEFAULT_ITERATIONS = 500
DEFAULT_GPU_MEM = 0.8
DEFAULT_SEED = 0
###


def _read_unpack(fmt, fh):
    return unpack(fmt, fh.read(calcsize(fmt)))


def _find_exe_dir():
    parent_dir = os.path.abspath(
        path_join(os.path.dirname(__file__), os.path.pardir))
    dir_to_exe = path_join(parent_dir, 'build', 'linux')
    exe_file = 't_sne_bhcuda'
    tsne_path = path_join(dir_to_exe, exe_file)

    return tsne_path


def t_sne(samples, use_scikit=False, files_dir=None, results_filename='result.dat', data_filename='data.dat',
          no_dims=DEFAULT_NO_DIMS, perplexity=DEFAULT_PERPLEXITY, theta=DEFAULT_THETA, eta=DEFAULT_ETA,
          iterations=DEFAULT_ITERATIONS, seed=DEFAULT_SEED, early_exaggeration=DEFAULT_EARLY_EXAGGERATION,
          gpu_mem=DEFAULT_GPU_MEM, randseed=DEFAULT_RANDOM_SEED, verbose=2):
    """
    Run t-sne on the sapplied samples (Nxsamples x Dfeatures array). It either:
    1) Calls the t_sne_bhcuda.exe (which should be in the Path of the OS somehow - maybe in the Scripts folder for
    Windows or the python/bin folder for Linux) which then runs t-sne either on the CPU or the GPU
    or 2) Calls the sklearn t-sne module (which runs only on CPU).

    Parameters
    ----------
    samples -- The N_examples X D_features array to t-sne
    use_scikit -- If yes use the sklearn t-sne implementation. Otherwise use the t_sne_bhcuda.exe
    files_dir -- The folder in which the t_sne_bhcuda.exe should look for the data_filename.dat and save the
    results_filename.dat
    results_filename -- The name of the file that the t_sne_bhcuda.exe saves the t-sne results in
    data_filename -- The name of the file that the t_sne_bhcuda.exe looks into for data to t-sne. This data file
    also has a header with all the parameters that the t_sne_bhcuda.exe needs to run.
    no_dims -- Number of dimensions of the t-sne embedding
    perplexity -- Defines the amount of samples whose distances are comparred to every sample (check sklearn and the
    van der Maatens paper)
    theta -- If > 0 then the algorithm run the burnes hat aproximation (with angle = theta). If = 0 then it runs the
    exact version. Values smaller than 0.2 do not add to much error.
    eta -- The learning rate
    iterations -- The number of itterations (usually around 1000 should suffice)
    early_exaggeration -- The amount by which the samples are initially pushed apart. Used only in the sckit-learn
    version
    seed -- Set to a number > 0 if the amount of samples is too large to t-sne. Then the algorithm will t-sne the first
    seed number of samples. Then it will compare the euclidean distance between every other sample and these t-sned
    samples. For each non t-sned sample it will find the 5 closest t-sned samples and place the new sample on the point
    of the t-sne space  that is given by the center of mass of those 5 closest ssamples. The mass of each closest
    sample is defined as the inverse of its euclidean distance to the new sample.
    gpu_mem -- If > 0 (and <= 1) then the t_sne_bhcuda.exe will run the eucledian distances calculations on the GPU
    (if possible) and will use (gpu_mem * 100) per cent of the available gpu memory to temporarily store results. If
    == 0 then the t_sne_bhcuda.exe will run only on the CPU. It has no affect if use_scikit = True
    randseed -- Set the random seed for the initiallization of the samples on the no_dims plane.
    verbose -- Define verbosity. 0 = No output, 1 = Basic output, 2 = Full output, 3 = Also save t-sne results in
    interim files after every iteration. Option 3 is used to save all steps of t-sne to explore the way the algorithm
    seperates the data (good for movies).

    Returns
    -------
    A N_examples X no_dims array of the embeded examples

    """
    if use_scikit:  # using python's scikit tsne implementation
        try:
            from sklearn.manifold import TSNE as tsne
        except ImportError:
            print('You do not have sklearn installed. Try calling t_sne with use_scikit=False'
                  'and gpu_mem=0 if you do not want to run the code in GPU.')
            return None
        if theta > 0:
            method = 'barnes_hut'
        elif theta == 0:
            method = 'exact'
        model = tsne(n_components=2, perplexity=perplexity, early_exaggeration=early_exaggeration,
                     learning_rate=eta, n_iter=iterations, n_iter_without_progress=iterations,
                     min_grad_norm=0, metric="euclidean", init="random", verbose=verbose,
                     random_state=None, method=method, angle=theta)
        t_sne_results = model.fit_transform(samples)

        return t_sne_results

    else:  # using the C++/cuda implementation
        save_data_for_tsne(samples, files_dir, data_filename, theta, perplexity,
                           eta, no_dims, iterations, seed, gpu_mem, verbose, randseed)
        del samples
        stdout = ''
        # Call t_sne_bhcuda and let it do its thing
        with Popen([_find_exe_dir(), ], cwd=files_dir, stdout=PIPE, bufsize=1, universal_newlines=True) \
                as t_sne_bhcuda_p:
            for line in iter(t_sne_bhcuda_p.stdout):
                print(line, end='')
                stdout += line
                sys.stdout.flush()
            t_sne_bhcuda_p.wait()
        print("TSNE return code: {}".format(t_sne_bhcuda_p.returncode))
        # if t_sne_bhcuda_p.returncode != 0:
        #    print('ERROR showing t_sne output:\n {}'.format(stdout))
        assert not t_sne_bhcuda_p.returncode, ('ERROR: Call to t_sne_bhcuda exited '
                                               'with a non-zero return code exit status, please ' +
                                               ('enable verbose mode and ' if not verbose else '') +
                                               'refer to the t_sne output for further details')

        return load_tsne_result(files_dir, results_filename)


def save_data_for_tsne(samples, files_dir, filename, theta, perplexity, eta, no_dims, iterations, seed, gpu_mem,
                       verbose, randseed):
    """
    Saves the samples array and all the rquired parameters in a filename file that the t_sne_bhcuda.exe can read

    Parameters
    ----------
    samples -- The N_examples X D_features array to t-sne
    files_dir -- The folder in which the t_sne_bhcuda.exe should look for the filename.dat
    filename -- The name of the file that the t_sne_bhcuda.exe looks into for data to t-sne. This data file
    also has a header with all the parameters that the t_sne_bhcuda.exe needs to run.
    theta -- If > 0 then the algorithm run the burnes hat aproximation (with angle = theta). If = 0 then it runs the
     exact version. Values smaller than 0.5 do not add to much error.
    perplexity -- Defines the amount of samples whose distances are comparred to every sample (check sklearn and the
    van der Maatens paper)
    eta -- The learning rate
    no_dims -- Number of dimensions of the t-sne embedding
    iterations -- The number of itterations (usually around 1000 should suffice)
    seed -- Set to a number > 0 if the amount of samples is too large to t-sne
    gpu_mem -- If > 0 (and <= 1) then the t_sne_bhcuda.exe will run the eucledian distances calculations on the GPU
    (if possible) and will use (gpu_mem * 100) per cent of the available gpu memory to temporarily store results. If
    == 0 then the t_sne_bhcuda.exe will run only on the CPU.
    verbose -- Define verbosity. 0 = No output, 1 = Basic output, 2 = Full output, 3 = Also save t-sne results in
    interim files after every iteration. Option 3 is used to save all steps of t-sne to explore the way the algorithm
    seperates the data (good for movies).
    randseed -- Set the random seed for the initiallization of the samples on the no_dims plane.

    Returns
    -------
    Nothing. Just creates the file with the header and its data
    """

    sample_dim = len(samples[0])
    sample_count = len(samples)
    os.makedirs(files_dir, exist_ok=True)
    with open(path_join(files_dir, filename), 'wb') as data_file:
        # Write the t_sne_bhcuda header
        data_file.write(pack('iidddiiifi', sample_count, sample_dim, theta, perplexity,
                             eta, no_dims, iterations, seed, gpu_mem, verbose))
        # Write the data
        for sample in samples:
            data_file.write(pack('{}d'.format(len(sample)), *sample))
        # Write random seed if specified
        if randseed != DEFAULT_RANDOM_SEED:
            data_file.write(pack('i', randseed))


def load_tsne_result(files_dir, filename):
    """
    Load the file that has the t_sne_bhcuda.exe saves the results into.

    Parameters
    ----------
    files_dir -- The folder in which the t_sne_bhcuda.exe should look for the filename.dat
    filename -- The name of the file that the t_sne_bhcuda.exe saves the t-sne results in

    Returns
    -------
    A N_examples X no_dims array of the embeded examples
    """
    t_sne_results = []
    # Read and pass on the results
    with open(path_join(files_dir, filename), 'rb') as output_file:
        # The first two integers are the number of samples and the dimensionality
        result_samples, result_dims = _read_unpack('ii', output_file)
        # Collect the results, but they may be out of order
        results = [_read_unpack('{}d'.format(result_dims), output_file)
                   for _ in range(result_samples)]
        # Now collect the landmark data so that we can return the data in
        #   the order it arrived
        results = [(_read_unpack('i', output_file), e) for e in results]
        # Put the results in order and yield it
        results.sort()
        for _, result in results:
            t_sne_results.append(result)
        return np.array(t_sne_results)
