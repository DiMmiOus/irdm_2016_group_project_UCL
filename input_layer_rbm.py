from __future__ import print_function
import numpy as np
import pandas as pd


class RBM:

    def __init__(self, num_visible, num_hidden, learning_rate = 0.1):

        self.num_hidden = num_hidden
        self.num_visible = num_visible
        self.learning_rate = learning_rate

        # Initialize a weight matrix, of dimensions (num_visible x num_hidden), using
        # a Gaussian distribution with mean 0 and standard deviation 0.1.
        self.weights = 0.1 * np.random.randn(self.num_visible, self.num_hidden)
        # Insert weights for the bias units into the first row and first column.
        self.weights = np.insert(self.weights, 0, 0, axis = 0)
        self.weights = np.insert(self.weights, 0, 0, axis = 1)

    def train(self, data, max_epochs=1000):
        """
        Train the machine.

        Parameters
        ----------
        data: A matrix where each row is a training example consisting of the states of visible units.
        max_epochs: Total number of epochs
        """

        num_examples = data.shape[0]

        unsoftmax = data

        # Softmax
        e = np.exp(data / 1.0)
        data = e / np.sum(e)

        # Insert bias units of 1 into the first column.
        data = np.insert(data, 0, 1, axis = 1)

        for epoch in range(max_epochs):
            # Clamp to the data and sample from the hidden units.
            # (This is the "positive CD phase", aka the reality phase.)
            pos_hidden_activations = np.dot(data, self.weights)
            pos_hidden_probs = self._logistic(pos_hidden_activations)
            pos_hidden_states = pos_hidden_probs > np.random.rand(num_examples, self.num_hidden + 1)
            # Note that we're using the activation *probabilities* of the hidden states, not the hidden states
            # themselves, when computing associations.

            pos_associations_i = np.zeros((num_examples, self.num_visible + 1, self.num_hidden + 1))

            # one visible x hidden pos_association per user
            # we are decomposing this matrix so we can distinguish each user's missing values
            for i in range(num_examples):

                pos_associations_i[i, :, :] = np.outer(data.T[:, i], pos_hidden_probs[i, :])

            # Reconstruct the visible units and sample again from the hidden units.
            # (This is the "negative CD phase", aka the daydreaming phase.)
            neg_visible_activations = np.dot(pos_hidden_states, self.weights.T)

            neg_visible_probs = self._logistic(neg_visible_activations)
            neg_visible_probs[:,0] = 1 # Fix the bias unit.
            neg_hidden_activations = np.dot(neg_visible_probs, self.weights)
            neg_hidden_probs = self._logistic(neg_hidden_activations)
            # Note, again, that we're using the activation *probabilities* when computing associations, not the states
            # themselves.

            neg_associations_i = np.zeros((num_examples, self.num_visible + 1, self.num_hidden + 1))

            for i in range(num_examples):

                neg_associations_i[i, :, :] = np.outer(neg_visible_probs.T[:, i], neg_hidden_probs[i, :])

            delta_wi = np.zeros((num_examples, self.num_visible + 1, self.num_hidden + 1))

            for i in range(0, num_examples):

                delta_wi[i, :, :] = pos_associations_i[i, :, :] - neg_associations_i[i, :, :]
                there = np.where(unsoftmax[i, :] == 0)[0]
                delta_wi[i, there, :] = 0.0

            # Update weights.
            self.weights += (self.learning_rate / num_examples) * np.sum(delta_wi, axis=0)
            # self.weights += self.learning_rate * ((pos_associations - neg_associations) / num_examples)

            error = np.sum((data - neg_visible_probs) ** 2)

            print("Epoch %s: error is %s" % (epoch, error))

    def run_visible(self, data):
        """
        Assuming the RBM has been trained (so that weights for the network have been learned),
        run the network on a set of visible units, to get a sample of the hidden units.

        Parameters
        ----------
        data: A matrix where each row consists of the states of the visible units.

        Returns
        -------
        hidden_states: A matrix where each row consists of the hidden units activated from the visible
        units in the data matrix passed in.
        """

        num_examples = data.shape[0]

        # Create a matrix, where each row is to be the hidden units (plus a bias unit)
        # sampled from a training example.
        hidden_states = np.ones((num_examples, self.num_hidden + 1))

        # Insert bias units of 1 into the first column of data.
        data = np.insert(data, 0, 1, axis = 1)

        # Calculate the activations of the hidden units.
        hidden_activations = np.dot(data, self.weights)
        # Calculate the probabilities of turning the hidden units on.
        hidden_probs = self._logistic(hidden_activations)
        # Turn the hidden units on with their specified probabilities.
        hidden_states[:,:] = hidden_probs > np.random.rand(num_examples, self.num_hidden + 1)
        # Always fix the bias unit to 1.
        # hidden_states[:,0] = 1

        # Ignore the bias units.
        hidden_states = hidden_states[:,1:]
        return hidden_states
      
    def _logistic(self, x):
        return 1.0 / (1 + np.exp(-x))

if __name__ == '__main__':

    # Read data from file
    r_cols = ['user_id', 'movie_id', 'rating', 'unix_timestamp']
    data_file = pd.read_table('u.data', sep='\t', names=r_cols, header=None)
    users = np.unique(data_file['user_id'])
    movies = np.unique(data_file['movie_id'])

    number_of_rows = len(users)
    number_of_columns = len(movies)
    movie_indices, user_indices = {}, {}

    for i in range(len(movies)):
        movie_indices[movies[i]] = i

    for i in range(len(users)):
        user_indices[users[i]] = i

    V = np.zeros((number_of_rows, number_of_columns))

    for line in data_file.values:
      u, i , r , gona = map(int,line)
      V[user_indices[u], movie_indices[i]] = r

    # Train input layer and save weights
    r = RBM(num_visible = number_of_columns, num_hidden = 50)
    training_data = V[0:750, :]
    r.train(training_data, max_epochs = 5000)
    np.savetxt('weights.out', r.weights, delimiter=',')
    # Get hidden states for training set
    np.savetxt('hidden.out', r.run_visible(V[0:750, :]), delimiter=',', fmt='%d')
