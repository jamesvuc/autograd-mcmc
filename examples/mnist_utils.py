import autograd.numpy as np

class Mnist:
    def __init__(self, fpath='/tmp/'):
        import os
        self._fpath = os.path.join(os.path.abspath(fpath), 'mnist.npz')

        if not os.path.exists(self._fpath):
            raise FileNotFoundError("""You need to download the mnist  dataset from  
                https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz""")
       
    def load_data(self):
        f = np.load(self._fpath)
        return (f['x_train'], f['y_train']), (f['x_test'], f['y_test'])

class Batches:
    def __init__(self, inputs, targets, batch_size = 64, shuffle=True):
        self._i = -1
        self.batch_size = batch_size
        self.num_batches = len(inputs) // batch_size

        if shuffle:
            randperm = np.arange(len(inputs)) ; np.random.shuffle(randperm)
            self._inputs = inputs[randperm]
            self._targets = targets[randperm]
        else:
            self._inputs = inputs
            self._targets = targets

    def next(self):
        self._i += 1
        idx = self._i % self.num_batches
        s = slice(idx * self.batch_size, (idx+1) * self.batch_size)
        return self._inputs[s], self._targets[s]

