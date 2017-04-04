# coding=utf-8

import numpy as np
from scipy.special import expit


class MLP:
    def __init__(self, layers, activations, cost_function="cross_entropy", learning_rate=0.5, momentum=0.9,
                 batch_size=100, init_wt=0.01):
        """
初始化FFN参数
        :param layers: 各层的单元数，以list形式传入
        :param activations: 隐藏层及输出层所使用的激活函数，以list形式传入，目前支持"linear"、"logistic"、"softmax"
        :param cost_function: 损失函数名称，目前支持"cross_entropy"
        :param learning_rate: 学习速率，默认为0.1
        :param momentum:默认为0.9
        :param batch_size:默认为100
        """

        self.bias = []
        #self.bias_gradient = []
        self.bias_delta = []
        self.weights = []
        #self.weights_gradient = []
        self.weights_delta = []
        self.layers = layers
        self.activations = activations
        self.cost_function = cost_function
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.batch_size = batch_size

        self.init_wt = init_wt
        self.y_to_mat = np.eye(self.layers[-1], dtype=np.int)

    def weights_bias_gradient_delta_init(self):
        """
根据layers初始化各层之间的weights参数
        """
        for i in range(len(self.layers) - 1):
            self.weights.append(self.init_wt*np.random.randn(self.layers[i], self.layers[i + 1]))
            #self.weights_gradient.append(np.zeros((self.layers[i], self.layers[i + 1])))
            self.weights_delta.append(np.zeros((self.layers[i], self.layers[i + 1])))
            self.bias.append(np.ones((self.layers[i + 1], 1)))
            #self.bias_gradient.append(np.zeros((self.layers[i + 1], 1)))
            self.bias_delta.append(np.zeros((self.layers[i + 1], 1)))


    def data_split(self, data, split=None):
        if not split:
            split = [0.6, 0.3, 0.1]
        np.random.shuffle(data)
        size = np.size(data, 0)
        [train, val_in, test] = np.split(data, [i * size for i in np.cumsum(split[:-1])])
        return train, val_in, test

    def fpro(self, data):
        out_layer_state = []
        out_layer_net_in = []
        layer_state = data
        out_layer_state.append(layer_state)
        for i in range(len(self.weights)):
            net_in = np.dot(self.weights[i].transpose(), layer_state)
            layer_state = MLP.__dict__[self.activations[i]](net_in + self.bias[i])
            out_layer_state.append(layer_state)
            out_layer_net_in.append(net_in)

        return out_layer_net_in,out_layer_state

    def logistic(net_in):
        return expit(net_in)

    def gradient_logistic(units_state):
        return units_state*(1.0-units_state)

    def softmax(net_in):
        max_in = np.max(net_in, axis=0)
        tmp1 = np.exp(net_in - max_in)
        tmp2 = np.sum(tmp1, axis=0)
        return tmp1 / tmp2

    def bp_deriv(self, units_state, target, net_ins):
        out = []
        target_mat = self.y_to_mat[:, target]
        out_layer_deriv = units_state[-1] - target_mat
        out.append(out_layer_deriv)
        deriv = out_layer_deriv
        for i in range(len(self.weights) - 1):
            deriv = np.dot(self.weights[-(i + 1)], deriv) * MLP.__dict__['gradient_'+self.activations[-(i+2)]](units_state[-(i+2)])
            #deriv = np.dot(self.weights[-(i + 1)], deriv)
            out.insert(0, deriv)
        return out

    def compute_gradient(self, deriv, states):
        weights_gradient = []
        bias_gradient = []
        for i in range(len(self.weights)):
            gradient_weights = np.dot(states[i], np.transpose(deriv[i])) * 1.0 / self.batch_size
            weights_gradient.append(gradient_weights)
            tmp = np.sum(deriv[i], 1) * 1.0 / self.batch_size
            gradient_bias = tmp[:, np.newaxis]
            bias_gradient.append(gradient_bias)
        return weights_gradient, bias_gradient

    def weights_update(self, weights_gradient, bias_gradient):
        for i in range(len(self.weights)):
            self.weights_delta[i] = self.weights_delta[i] * self.momentum - \
                                    weights_gradient[i]
            self.bias_delta[i] = self.bias_delta[i] * self.momentum - bias_gradient[i]
            self.weights[i] += self.weights_delta[i] * self.learning_rate
            self.bias[i] = self.bias[i] + self.bias_delta[i] * self.learning_rate

    def loss_cal(self, x, y):
        y = np.asarray(y, dtype=np.int)
        out = self.predict(x.T)
        loss = -np.sum(self.y_to_mat[:, y] * np.log(out + 1e-30)) / len(y)
        return loss

    def train(self, train, val_in, test, epochs=7, show_train_error_after=100, show_val_error_after=1000):
        self.weights_bias_gradient_delta_init()
        trunk_loss = 0.0

        while epochs:
            batch_complete = 0
            batch_count = 0
            trainset_loss = 0
            np.random.shuffle(train)
            train_x = train[:, :-1].T
            train_y = train[:, -1]
            batch_total = int(np.size(train_x, axis=1) / self.batch_size)
            for i in range(batch_total):
                weights_gradient = []
                bias_gradient = []
                x = train_x[:, i * self.batch_size:(i + 1) * self.batch_size]
                y = np.asarray(train_y[i * self.batch_size:(i + 1) * self.batch_size], dtype=np.int)
                units_net_in, units_state = self.fpro(x)
                bp_deriv = self.bp_deriv(units_state, y, units_net_in)
                weights_gradient, bias_gradient = self.compute_gradient(bp_deriv, units_state)
                self.weights_update(weights_gradient, bias_gradient)
                batch_complete += 1
                batch_count += 1
                train_loss = -np.sum(self.y_to_mat[:, y] * np.log(units_state[-1] + 1e-30))
                trunk_loss += (train_loss - trunk_loss) / batch_count
                trainset_loss += (train_loss - trainset_loss) / batch_complete
                if batch_count == show_train_error_after:
                    print("Batch:%d, Train error:%.3f" % (batch_complete, trunk_loss))
                    y_est = np.argmax(units_state[-1], axis=0)
                    percent_right = np.sum(y_est==y)*1.0/np.size(y)
                    print("Percent right: %.3f"%percent_right)
                    batch_count = 0
                    trunk_loss = 0.0
                if batch_complete % show_val_error_after == 0:
                    val_loss = self.loss_cal(val_in[:, :-1].T, val_in[:, -1])
                    print("Validation error: %.3f" % val_loss)
                    val_predict = self.predict_precise(val_in[:, :-1].T, val_in[:, -1])
                    print("Validation right: %.3f" %val_predict)
            print("Average train error: %.3f" % trainset_loss)
            epochs -= 1
        #print("Final train error: %.3f" % trunk_loss)
        val_loss = self.loss_cal(val_in[:, :-1].T, val_in[:, -1])
        print("Final validation error: %.3f" % val_loss)
        test_loss = self.loss_cal(test[:, :-1].T, test[:, -1])
        print("Test error: %.3f" % test_loss)
        test_predict = self.predict_precise(test[:, :-1].T, test[:, -1])
        print("Test right: %.3f" % test_predict)

    def predict_precise(self, x, y):
        x = np.array(x)
        layer_state = x
        for i in range(len(self.weights)):
            net_in = np.dot(self.weights[i].transpose(), layer_state) + self.bias[i]
            layer_state = MLP.__dict__[self.activations[i]](net_in)
        y_predict = np.argmax(layer_state, axis=0)
        pred_percent = np.sum(y_predict==y)/np.size(y)
        return pred_percent

    def predict(self, x):
        x = np.array(x)
        layer_state = x.T
        for i in range(len(self.weights)):
            net_in = np.dot(self.weights[i].transpose(), layer_state) + self.bias[i]
            layer_state = MLP.__dict__[self.activations[i]](net_in)
        return layer_state





if __name__ == "__main__":
    from sklearn.datasets import load_digits
    import numpy as np

    digits = load_digits()
    data = np.hstack((digits.data, digits.target[:, np.newaxis]))
    nn = MLP(layers=[64,  100,  10], activations=[ 'logistic', 'logistic',  'softmax'])
    train_data, val_data, test_data = nn.data_split(data=data, split=[0.8, 0.1, 0.1])
    nn.train(train=train_data, val_in=val_data, test=test_data,
             epochs=300, show_train_error_after=1 ,show_val_error_after=3)
