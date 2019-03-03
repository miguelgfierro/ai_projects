"""
Replicate "Very Deep Convolutional Networks for Natural Language Processing" by Alexis Conneau,
Holger Schwenk, Yann Le Cun, Loic Barraeau, 2016

New NLP architecture:
1. Operate at lowest atomic representation of text (characters)
2. Use deep-stack of local operations to learn high-level hierarchical representation

attribution: https://github.com/ilkarman/NLP-Sentiment/
"""

import numpy as np
import pandas as pd
import mxnet as mx
import wget
import time
import functools
import threading
import os.path
import Queue
import pickle
from mxnet.io import DataBatch

ctx = mx.gpu(0)
AZ_ACC = "amazonsentimenik"
AZ_CONTAINER = "textclassificationdatasets"
ALPHABET = list("abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:'\"/\\|_@#$%^&*~`+ =<>()[]{}")
FEATURE_LEN = 1014
BATCH_SIZE = 128
EPOCHS = 10
SD = 0.05  # std for gaussian distribution
NOUTPUT = 2
DATA_SHAPE = (BATCH_SIZE, 1, FEATURE_LEN, 1)


def download_file(url):
    # Create file-name
    local_filename = url.split('/')[-1]
    if os.path.isfile(local_filename):
        pass
        # print("The file %s already exist in the current directory\n" % local_filename)
    else:
        # Download
        print("downloading ...\n")
        wget.download(url)
        print('saved data\n')


def load_file(infile):
    """
    Takes .csv and returns loaded data along with labels
    """
    print("processing data frame: %s" % infile)
    # Get data from windows blob
    download_file('https://%s.blob.core.windows.net/%s/%s' % (AZ_ACC, AZ_CONTAINER, infile))
    # load data into dataframe
    df = pd.read_csv(infile,
                     header=None,
                     names=['sentiment', 'summary', 'text'])
    # concat summary, review; trim to 1014 char; reverse; lower
    df['rev'] = df.apply(lambda x: "%s %s" % (x['summary'], x['text']), axis=1)
    df.rev = df.rev.str[:FEATURE_LEN].str[::-1].str.lower()
    # store class as nparray
    df.sentiment -= 1
    y_split = np.asarray(df.sentiment, dtype='bool')
    print("finished processing data frame: %s" % infile)
    print("data contains %d obs, each epoch will contain %d batches" % (df.shape[0], df.shape[0] // BATCH_SIZE))
    return df.rev, y_split


def load_data_frame(X_data, y_data, batch_size=128, shuffle=False):
    """
    For low RAM this methods allows us to keep only the original data
    in RAM and calculate the features (which are orders of magnitude bigger
    on the fly). This keeps only 10 batches worth of features in RAM using
    asynchronous programing and yields one DataBatch() at a time.
    """

    if shuffle:
        idx = X_data.index
        assert len(idx) == len(y_data)
        rnd = np.random.permutation(idx)
        X_data = X_data.reindex(rnd)
        y_data = y_data[rnd]

    # Dictionary to create character vectors
    char_index = dict((c, i + 2) for i, c in enumerate(ALPHABET))

    # Yield processed batches asynchronously
    # Buffy 'batches' at a time
    def async_prefetch_wrp(iterable, buffy=30):
        poison_pill = object()

        def worker(q, it):
            for item in it:
                q.put(item)
            q.put(poison_pill)

        queue = Queue.Queue(buffy)
        it = iter(iterable)
        thread = threading.Thread(target=worker, args=(queue, it))
        thread.daemon = True
        thread.start()
        while True:
            item = queue.get()
            if item == poison_pill:
                return
            else:
                yield item

    # Async wrapper around
    def async_prefetch(func):
        @functools.wraps(func)
        def wrapper(*args, **kwds):
            return async_prefetch_wrp(func(*args, **kwds))

        return wrapper

    @async_prefetch
    def feature_extractor(dta, val):
        # Yield mini-batch amount of character vectors
        # X_split = np.zeros([batch_size, 1, FEATURE_LEN, len(ALPHABET)], dtype='bool')
        X_split = np.zeros([batch_size, 1, FEATURE_LEN, 1], dtype='int')
        for ti, tx in enumerate(dta):
            chars = list(tx)
            for ci, ch in enumerate(chars):
                if ch in ALPHABET:
                    X_split[ti % batch_size][0][ci] = char_index[ch]
                    # X_split[ti % batch_size][0][ci] = np.array(character_hash[ch], dtype='bool')

            # No padding -> only complete batches processed
            if (ti + 1) % batch_size == 0:
                yield mx.nd.array(X_split), mx.nd.array(val[ti + 1 - batch_size:ti + 1])
                # X_split = np.zeros([batch_size, 1, FEATURE_LEN, len(ALPHABET)], dtype='bool')
                X_split = np.zeros([batch_size, 1, FEATURE_LEN, 1], dtype='int')

    # Yield one mini-batch at a time and asynchronously process to keep 4 in queue
    for Xsplit, ysplit in feature_extractor(X_data, y_data):
        yield DataBatch(data=[Xsplit], label=[ysplit])


class k_max_pool(mx.operator.CustomOp):

    """
    https://github.com/CNevd/DeepLearning-Mxnet/blob/master/DCNN/dcnn_train.py#L15
    """

    def __init__(self, k):
        super(k_max_pool, self).__init__()
        self.k = int(k)

    def forward(self, is_train, req, in_data, out_data, aux):
        x = in_data[0].asnumpy()
        # assert(4 == len(x.shape))
        ind = np.argsort(x, axis=2)
        sorted_ind = np.sort(ind[:, :, -(self.k):, :], axis=2)
        dim0, dim1, dim2, dim3 = sorted_ind.shape
        self.indices_dim0 = np.arange(dim0).repeat(dim1 * dim2 * dim3)
        self.indices_dim1 = np.transpose(
            np.arange(dim1).repeat(dim2 * dim3).reshape((dim1 * dim2 * dim3, 1)).repeat(dim0, axis=1)).flatten()
        self.indices_dim2 = sorted_ind.flatten()
        self.indices_dim3 = np.transpose(
            np.arange(dim3).repeat(dim2).reshape((dim2 * dim3, 1)).repeat(dim0 * dim1, axis=1)).flatten()
        y = x[self.indices_dim0, self.indices_dim1, self.indices_dim2, self.indices_dim3].reshape(sorted_ind.shape)
        self.assign(out_data[0], req[0], mx.nd.array(y))

    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
        x = out_grad[0].asnumpy()
        y = in_data[0].asnumpy()
        # assert(4 == len(x.shape))
        # assert(4 == len(y.shape))
        y[:, :, :, :] = 0
        y[self.indices_dim0, self.indices_dim1, self.indices_dim2, self.indices_dim3] \
            = x.reshape([x.shape[0] * x.shape[1] * x.shape[2] * x.shape[3], ])
        self.assign(in_grad[0], req[0], mx.nd.array(y))


@mx.operator.register("k_max_pool")
class k_max_poolProp(mx.operator.CustomOpProp):
    def __init__(self, k):
        self.k = int(k)
        super(k_max_poolProp, self).__init__(True)

    def list_argument(self):
        return ['data']

    def list_outputs(self):
        return ['output']

    def infer_shape(self, in_shape):
        data_shape = in_shape[0]
        assert (len(data_shape) == 4)
        out_shape = (data_shape[0], data_shape[1], self.k, data_shape[3])
        return [data_shape], [out_shape]

    def create_operator(self, ctx, shapes, dtypes):
        return k_max_pool(self.k)


def create_vdcnn():
    """
    29 Convolutional Layers

    We want to increase the number of conv layers to 29 in the following structure:
    1 | 10 | 10 | 4 | 4 -> 4.6 million params

    We down-sample using convolutions with stride=2

    ToDo:
    2. Temporal batch norm vs. batch norm? -> "Temp batch norm applies same kind of regularization
    as batch norm, except that the activations in a mini-batch are jointly normalized over temporal
    instead of spatial locations"
    3. Double check that optional shortcuts are not used for the smaller nets (only for 49 conv layer one,
    as they reduce performance for 9, 17, 29 conv. layer models)
    """

    vocab_size = 69
    embedding_size = 16
    temp_kernel = (3, embedding_size)
    kernel = (3, 1)
    stride = (2, 1)
    padding = (1, 0)
    kmax = 8
    num_filters1 = 64
    num_filters2 = 128
    num_filters3 = 256
    num_filters4 = 512

    input_x = mx.sym.Variable('data')  # placeholder for input
    input_y = mx.sym.Variable('softmax_label')  # placeholder for output

    # Lookup Table 16
    embed_layer = mx.symbol.Embedding(
        data=input_x, input_dim=vocab_size, output_dim=embedding_size, name='word_embedding')
    embed_out = mx.sym.Reshape(
        data=embed_layer, shape=(BATCH_SIZE, 1, FEATURE_LEN, embedding_size))

    # Temp Conv (in: batch, 1, 1014, 16)
    conv0 = mx.symbol.Convolution(
        data=embed_out, kernel=temp_kernel, pad=padding, num_filter=num_filters1)
    act0 = mx.symbol.Activation(
        data=conv0, act_type='relu')

    # CONVOLUTION_BLOCK (1 of 4) -> 64 FILTERS
    # 10 Convolutional Layers
    conv11 = mx.symbol.Convolution(
        data=act0, kernel=kernel, pad=padding, num_filter=num_filters1)
    norm11 = mx.symbol.BatchNorm(
        data=conv11)
    act11 = mx.symbol.Activation(
        data=norm11, act_type='relu')
    conv12 = mx.symbol.Convolution(
        data=act11, kernel=kernel, pad=padding, num_filter=num_filters1)
    norm12 = mx.symbol.BatchNorm(
        data=conv12)
    act12 = mx.symbol.Activation(
        data=norm12, act_type='relu')

    conv21 = mx.symbol.Convolution(
        data=act12, kernel=kernel, pad=padding, num_filter=num_filters1)
    norm21 = mx.symbol.BatchNorm(
        data=conv21)
    act21 = mx.symbol.Activation(
        data=norm21, act_type='relu')
    conv22 = mx.symbol.Convolution(
        data=act21, kernel=kernel, pad=padding, num_filter=num_filters1)
    norm22 = mx.symbol.BatchNorm(
        data=conv22)
    act22 = mx.symbol.Activation(
        data=norm22, act_type='relu')

    conv31 = mx.symbol.Convolution(
        data=act22, kernel=kernel, pad=padding, num_filter=num_filters1)
    norm31 = mx.symbol.BatchNorm(
        data=conv31)
    act31 = mx.symbol.Activation(
        data=norm31, act_type='relu')
    conv32 = mx.symbol.Convolution(
        data=act31, kernel=kernel, pad=padding, num_filter=num_filters1)
    norm32 = mx.symbol.BatchNorm(
        data=conv32)
    act32 = mx.symbol.Activation(
        data=norm32, act_type='relu')

    conv41 = mx.symbol.Convolution(
        data=act32, kernel=kernel, pad=padding, num_filter=num_filters1)
    norm41 = mx.symbol.BatchNorm(
        data=conv41)
    act41 = mx.symbol.Activation(
        data=norm41, act_type='relu')
    conv42 = mx.symbol.Convolution(
        data=act41, kernel=kernel, pad=padding, num_filter=num_filters1)
    norm42 = mx.symbol.BatchNorm(
        data=conv42)
    act42 = mx.symbol.Activation(
        data=norm42, act_type='relu')

    conv51 = mx.symbol.Convolution(
        data=act42, kernel=kernel, pad=padding, num_filter=num_filters1)
    norm51 = mx.symbol.BatchNorm(
        data=conv51)
    act51 = mx.symbol.Activation(
        data=norm51, act_type='relu')
    conv52 = mx.symbol.Convolution(
        data=act51, kernel=kernel, pad=padding, num_filter=num_filters1)
    norm52 = mx.symbol.BatchNorm(
        data=conv52)
    act52 = mx.symbol.Activation(
        data=norm52, act_type='relu')

    # CONVOLUTION_BLOCK (2 of 4) -> 128 FILTERS
    # 10 Convolutional Layers

    # First down-sampling
    conv61 = mx.symbol.Convolution(
        data=act52, kernel=kernel, pad=padding, stride=stride, num_filter=num_filters2)

    norm61 = mx.symbol.BatchNorm(
        data=conv61)
    act61 = mx.symbol.Activation(
        data=norm61, act_type='relu')
    conv62 = mx.symbol.Convolution(
        data=act61, kernel=kernel, pad=padding, num_filter=num_filters2)
    norm62 = mx.symbol.BatchNorm(
        data=conv62)
    act62 = mx.symbol.Activation(
        data=norm62, act_type='relu')

    conv71 = mx.symbol.Convolution(
        data=act62, kernel=kernel, pad=padding, num_filter=num_filters2)
    norm71 = mx.symbol.BatchNorm(
        data=conv71)
    act71 = mx.symbol.Activation(
        data=norm71, act_type='relu')
    conv72 = mx.symbol.Convolution(
        data=act71, kernel=kernel, pad=padding, num_filter=num_filters2)
    norm72 = mx.symbol.BatchNorm(
        data=conv72)
    act72 = mx.symbol.Activation(
        data=norm72, act_type='relu')

    conv81 = mx.symbol.Convolution(
        data=act72, kernel=kernel, pad=padding, num_filter=num_filters2)
    norm81 = mx.symbol.BatchNorm(
        data=conv81)
    act81 = mx.symbol.Activation(
        data=norm81, act_type='relu')
    conv82 = mx.symbol.Convolution(
        data=act81, kernel=kernel, pad=padding, num_filter=num_filters2)
    norm82 = mx.symbol.BatchNorm(
        data=conv82)
    act82 = mx.symbol.Activation(
        data=norm82, act_type='relu')

    conv91 = mx.symbol.Convolution(
        data=act82, kernel=kernel, pad=padding, num_filter=num_filters2)
    norm91 = mx.symbol.BatchNorm(
        data=conv91)
    act91 = mx.symbol.Activation(
        data=norm91, act_type='relu')
    conv92 = mx.symbol.Convolution(
        data=act91, kernel=kernel, pad=padding, num_filter=num_filters2)
    norm92 = mx.symbol.BatchNorm(
        data=conv92)
    act92 = mx.symbol.Activation(
        data=norm92, act_type='relu')

    conv101 = mx.symbol.Convolution(
        data=act92, kernel=kernel, pad=padding, num_filter=num_filters2)
    norm101 = mx.symbol.BatchNorm(
        data=conv101)
    act101 = mx.symbol.Activation(
        data=norm101, act_type='relu')
    conv102 = mx.symbol.Convolution(
        data=act101, kernel=kernel, pad=padding, num_filter=num_filters2)
    norm102 = mx.symbol.BatchNorm(
        data=conv102)
    act102 = mx.symbol.Activation(
        data=norm102, act_type='relu')


    # CONVOLUTION_BLOCK (3 of 4) -> 256 FILTERS
    # 4 Convolutional Layers

    # Second down-sampling
    conv111 = mx.symbol.Convolution(
        data=act102, kernel=kernel, pad=padding, stride=stride, num_filter=num_filters3)

    norm111 = mx.symbol.BatchNorm(
        data=conv111)
    act111 = mx.symbol.Activation(
        data=norm111, act_type='relu')
    conv112 = mx.symbol.Convolution(
        data=act111, kernel=kernel, pad=padding, num_filter=num_filters3)
    norm112 = mx.symbol.BatchNorm(
        data=conv112)
    act112 = mx.symbol.Activation(
        data=norm112, act_type='relu')

    conv121 = mx.symbol.Convolution(
        data=act112, kernel=kernel, pad=padding, num_filter=num_filters3)
    norm121 = mx.symbol.BatchNorm(
        data=conv121)
    act121 = mx.symbol.Activation(
        data=norm121, act_type='relu')
    conv122 = mx.symbol.Convolution(
        data=act121, kernel=kernel, pad=padding, num_filter=num_filters3)
    norm122 = mx.symbol.BatchNorm(
        data=conv122)
    act122 = mx.symbol.Activation(
        data=norm122, act_type='relu')

    # CONVOLUTION_BLOCK (4 of 4) -> 512 FILTERS
    # 4 Convolutional Layers

    # Third down-sampling
    conv131 = mx.symbol.Convolution(
        data=act122, kernel=kernel, pad=padding, stride=stride, num_filter=num_filters4)

    norm131 = mx.symbol.BatchNorm(
        data=conv131)
    act131 = mx.symbol.Activation(
        data=norm131, act_type='relu')
    conv132 = mx.symbol.Convolution(
        data=act131, kernel=kernel, pad=padding, num_filter=num_filters4)
    norm132 = mx.symbol.BatchNorm(
        data=conv132)
    act132 = mx.symbol.Activation(
        data=norm132, act_type='relu')

    conv141 = mx.symbol.Convolution(
        data=act132, kernel=kernel, pad=padding, num_filter=num_filters4)
    norm141 = mx.symbol.BatchNorm(
        data=conv141)
    act141 = mx.symbol.Activation(
        data=norm141, act_type='relu')
    conv142 = mx.symbol.Convolution(
        data=act141, kernel=kernel, pad=padding, num_filter=num_filters4)
    norm142 = mx.symbol.BatchNorm(
        data=conv142)
    act142 = mx.symbol.Activation(
        data=norm142, act_type='relu')

    # K-max pooling (k=8)
    kpool = mx.symbol.Custom(
        data=act142, op_type='k_max_pool', k=kmax)

    # Flatten (dimensions * feature length * filters)
    flatten = mx.symbol.Flatten(data=kpool)

    # First fully connected
    fc1 = mx.symbol.FullyConnected(
        data=flatten, num_hidden=4096)
    act_fc1 = mx.symbol.Activation(
        data=fc1, act_type='relu')
    # Second fully connected
    fc2 = mx.symbol.FullyConnected(
        data=act_fc1, num_hidden=2048)
    act_fc2 = mx.symbol.Activation(
        data=fc2, act_type='relu')
    # Third fully connected
    fc3 = mx.symbol.FullyConnected(
        data=act_fc2, num_hidden=NOUTPUT)
    net = mx.symbol.SoftmaxOutput(
        data=fc3, label=input_y, name="softmax")

    #Debug:
    arg_shape, output_shape, aux_shape = net.infer_shape(data=(DATA_SHAPE))
    print("Arg Shape: ", arg_shape)
    print("Output Shape: ", output_shape)
    print("Aux Shape: ", aux_shape)
    print("Created network")

    return net


def save_check_point(mod_arg, mod_aux, pre, epoch):
    """
    Save model each epoch, load as:

    sym, arg_params, aux_params = \
        mx.model.load_checkpoint(model_prefix, n_epoch_load)

    # assign parameters
    mod.set_params(arg_params, aux_params)

    OR

    mod.fit(..., arg_params=arg_params, aux_params=aux_params,
            begin_epoch=n_epoch_load)
    """

    save_dict = {('arg:%s' % k): v for k, v in mod_arg.items()}
    save_dict.update({('aux:%s' % k): v for k, v in mod_aux.items()})
    param_name = '%s-%04d.pk' % (pre, epoch)
    pickle.dump(save_dict, open(param_name, "wb"))
    print('Saved checkpoint to \"%s\"' % param_name)


def load_check_point(file_name):
    # Load file
    print(file_name)
    save_dict = pickle.load(open(file_name, "rb"))
    # Extract data from save
    arg_params = {}
    aux_params = {}
    for k, v in save_dict.items():
        tp, name = k.split(':', 1)
        if tp == 'arg':
            arg_params[name] = v
        if tp == 'aux':
            aux_params[name] = v

    # Recreate model
    cnn = create_vdcnn()
    mod = mx.mod.Module(cnn, context=ctx)

    # Bind shape
    mod.bind(data_shapes=[('data', DATA_SHAPE)],
             label_shapes=[('softmax_label', (BATCH_SIZE,))])

    # assign parameters from save
    mod.set_params(arg_params, aux_params)
    print('Model loaded from disk')

    return mod


def train_model(train_fname):
    # Create mx.mod.Module()
    cnn = create_vdcnn()
    mod = mx.mod.Module(cnn, context=ctx)

    # Bind shape
    mod.bind(data_shapes=[('data', DATA_SHAPE)],
             label_shapes=[('softmax_label', (BATCH_SIZE,))])

    # Initialise parameters and optimiser
    mod.init_params(mx.init.Normal(sigma=SD))
    mod.init_optimizer(optimizer='sgd',
                       optimizer_params={
                           "learning_rate": 0.01,
                           "momentum": 0.9,
                           "wd": 0.00001,
                           "rescale_grad": 1.0 / BATCH_SIZE
                       })

    # Load Data
    X_train, y_train = load_file('amazon_review_polarity_train.csv')

    # Train
    print("Alphabet %d characters: " % len(ALPHABET), ALPHABET)
    print("started training")
    tic = time.time()

    # Evaluation metric:
    metric = mx.metric.Accuracy()

    # Train EPOCHS
    for epoch in range(EPOCHS):
        t = 0
        metric.reset()
        tic_in = time.time()
        for batch in load_data_frame(X_data=X_train,
                                     y_data=y_train,
                                     batch_size=BATCH_SIZE,
                                     shuffle=True):
            # Push data forwards and update metric
            mod.forward_backward(batch)
            mod.update()
            mod.update_metric(metric, batch.label)

            # For training + testing
            # mod.forward(batch, is_train=True)
            # mod.update_metric(metric, batch.label)
            # Get weights and update
            # For training only
            # mod.backward()
            # mod.update()
            # Log every 50 batches = 128*50 = 6400
            t += 1
            if t % 50 == 0:
                train_t = time.time() - tic_in
                metric_m, metric_v = metric.get()
                print("epoch: %d iter: %d metric(%s): %.4f dur: %.0f" % (epoch, t, metric_m, metric_v, train_t))

        # Checkpoint
        arg_params, aux_params = mod.get_params()
        save_check_point(mod_arg=arg_params,
                         mod_aux=aux_params,
                         pre=train_fname,
                         epoch=epoch)
        print("Finished epoch %d" % epoch)

    print("Done. Finished in %.0f seconds" % (time.time() - tic))


def test_model(test_fname):
    # Load saved model:
    mod = load_check_point(test_fname)
    # assert mod.binded and mod.params_initialized

    # Load data
    X_test, y_test = load_file('amazon_review_polarity_test.csv')

    # Score accuracy
    metric = mx.metric.Accuracy()

    # Test batches
    for batch in load_data_frame(X_data=X_test,
                                 y_data=y_test,
                                 batch_size=len(y_test)):
        mod.forward(batch, is_train=False)
        mod.update_metric(metric, batch.label)

        metric_m, metric_v = metric.get()
        print("TEST(%s): %.4f" % (metric_m, metric_v))


if __name__ == '__main__':

    # Train to 10 epochs
    train_model('v2_vdcnn_amazon_adv')

    # Load trained and test
    test_model('v2_vdcnn_amazon_adv-0009.pk')
