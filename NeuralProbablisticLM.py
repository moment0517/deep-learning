import numpy as np
from FeedForwardNN import MLP

class NPLM(MLP):
    def __init__(self, n_grams, vocab_size, distributed_units, hidden_units,
               learning_rate=0.1, momentum=0.9, batch_size=100, init_wt=1):
        self.n_gram = n_grams
        self.init_wt = init_wt
        MLP.__init__(self,
                     layers=[distributed_units*(n_grams-1), hidden_units,vocab_size],
                     activations=['logistic', 'softmax'],
                     learning_rate=learning_rate,
                     momentum=momentum,
                     batch_size=batch_size)
        self.vocab_size = vocab_size
        self.distributed_units = distributed_units
        self.embedding_weights = np.zeros((vocab_size, distributed_units))
        self.embedding_weights_delta = np.zeros((vocab_size, distributed_units))
        self.y_to_mat = np.eye(vocab_size, dtype=np.int)

    def train(self, train_data, val_data, test_data, vocab, epochs=5, show_train_error_after=100, show_val_error_after=1000):
        self.vocab = vocab
        self.weights_bias_gradient_delta_init()
        self.embedding_weights = self.init_wt * np.random.randn(self.vocab_size, self.distributed_units)
        trunk_loss = 0.0

        for epoch in range(epochs):
            batch_complete = 0
            batch_count = 0
            trainset_loss = 0
            np.random.shuffle(train_data)
            train_x = train_data[:, :-1].T - 1
            train_y = train_data[:, -1] - 1
            batch_total = int(np.size(train_x, axis=1) / self.batch_size)
            for i in range(batch_total):
                x = train_x[:, i * self.batch_size:(i + 1) * self.batch_size]
                y = np.asarray(train_y[i * self.batch_size:(i + 1) * self.batch_size], dtype=np.int)
                x_distributed = self.embedding_weights[x.T].reshape(-1, self.layers[0])
                x_distributed = x_distributed.T
                units_net_in, units_state = self.fpro(x_distributed)
                bp_deriv = self.bp_deriv(units_state, y, units_net_in)
                dist_deriv = np.dot(self.weights[0], bp_deriv[0])
                weights_gradient, bias_gradient = self.compute_gradient(bp_deriv, units_state)
                embedding_weights_gradient = np.zeros((self.vocab_size, self.distributed_units))
                for i in range(self.n_gram-1):
                    tmp =  np.dot(self.y_to_mat[:,x[i,:]], dist_deriv[i*self.distributed_units:(i+1)*self.distributed_units, :].T)/self.batch_size
                    embedding_weights_gradient = embedding_weights_gradient + tmp
                #embedding_weights_gradient = embedding_weights_gradient*1.0/self.n_gram
                self.weights_update(weights_gradient, bias_gradient)
                self.embedding_weights_delta = self.embedding_weights_delta*self.momentum + embedding_weights_gradient
                self.embedding_weights -= self.embedding_weights_delta*self.learning_rate
                batch_complete += 1
                batch_count += 1
                train_loss = -np.sum(self.y_to_mat[:, y] * np.log(units_state[-1] + 1e-30)) / self.batch_size
                trunk_loss += (train_loss - trunk_loss) / batch_count
                trainset_loss += (train_loss - trainset_loss) / batch_complete
                if batch_count == show_train_error_after:
                    print("Epoch %d Batch:%d, Train error:%.3f" % (epoch+1, batch_complete, trunk_loss))
                    y_est = np.argmax(units_state[-1], axis=0)
                    percent_right = np.sum(y_est==y)*1.0/np.size(y)
                    print("Percent right: %.3f"%percent_right)
                    batch_count = 0
                    trunk_loss = 0.0
                if batch_complete % show_val_error_after == 0:
                    val_x = val_data[:, :-1].T - 1
                    val_y = val_data[:, -1] - 1
                    val_x_distributed = self.embedding_weights[val_x.T].reshape(-1, self.distributed_units*(self.n_gram-1)).T
                    val_loss = self.loss_cal(val_x_distributed, val_y)
                    print("Validation error: %.3f" % val_loss)
                    val_predict = self.predict_precise(val_x_distributed, val_y)
                    print("Validation right: %.3f" %val_predict)
            print("Average train error: %.3f" % trainset_loss)
            test_x = test_data[:, :-1].T - 1
            test_y = test_data[:, -1] - 1
            test_x_distributed = self.embedding_weights[test_x.T].reshape(-1, self.distributed_units*(self.n_gram-1)).T
            test_loss = self.loss_cal(test_x_distributed, test_y)
            print("Test error: %.3f" % test_loss)
            test_predict = self.predict_precise(test_x_distributed, test_y)
            print("Test right: %.3f" % test_predict)

    def predict_next_word(self, context, num_predictions):
        x = []
        for i in context:
            if i.encode() not in self.vocab:
                print("%s is not int vocab"%i)
                return 0
            x.append(self.vocab.index(i.encode()))
        x = np.array(x, dtype=np.int)
        layer_state = self.embedding_weights[x].reshape(-1, self.distributed_units*(self.n_gram-1)).T
        for i in range(len(self.weights)):
            net_in = np.dot(self.weights[i].transpose(), layer_state) + self.bias[i]
            layer_state = MLP.__dict__[self.activations[i]](net_in)
        sort_pro = np.argsort(layer_state,axis=0)
        for i in range(num_predictions):
            print(context, "%s %.3f" % (self.vocab[sort_pro[249-i]].decode(), layer_state[sort_pro[249-i]]))

if __name__ == '__main__':
    import numpy as np

    train_data = np.genfromtxt('./neural language mode/trainData.txt', dtype="int")
    val_data = np.genfromtxt('./neural language mode/validationData.txt', dtype="int")
    test_data = np.genfromtxt('./neural language mode/testData.txt', dtype="int")
    vocab = np.genfromtxt('./neural language mode/vocab.txt', dtype='S10')
    vocab = tuple(vocab)
    nn = NPLM(n_grams=4, vocab_size=250, distributed_units=50, hidden_units=200, momentum=0.9, learning_rate=0.5)
    nn.train(train_data.T, val_data.T, test_data.T, vocab, epochs=1)
    nn.predict_next_word(("he","want", "to"), 3)






