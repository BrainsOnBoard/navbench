import numpy as np


class InfoMax:
    def __init__(self, num_inputs, num_hidden=None, learning_rate=0.000001, seed=None):
        self.learning_rate = learning_rate

        # seed may be None, in which case it'll be initialised by platform
        np.random.seed(seed)

        # Save seed in case we want it again
        self.seed = np.random.get_state()[1][0]
        print('Seed for InfoMax net: %i' % self.seed)

        if num_hidden is None:
            num_hidden = num_inputs

        weights = np.random.randn(num_inputs, num_hidden)

        # Normalise so that weights.mean() is approx 0 and weights.std() is approx 1
        weights -= weights.mean()
        weights /= weights.std()
        self.weights = weights

    def train(self, image):
        u = self.weights * image.ravel()
        y = np.tanh(u)
        lrate = self.learning_rate / u.shape[0]
        val = (np.eye(self.weights.shape[0]) - (y + u)
            * np.transpose(u) * self.weights)
        self.weights += lrate * val
        assert not np.isnan(self.weights).any()

    def test(self, image):
        return np.sum(np.abs(self.weights * image.ravel()))

    def ridf(self, image, step=1):
        vals = []
        for rot in range(0, image.shape[1], step):
            rot_image = np.roll(image, rot, axis=1)
            vals.append(self.test(rot_image))
        return np.array(vals)
