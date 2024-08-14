import numpy as np
class Model:
    def __init__(self):

        self.w1 = np.random.randn(1000, 9) * .01
        self.w2 = np.random.randn(1, 1000) * .01

    def predict(self, inputs):
        x = inputs

        Z_1 = self.w1 @ x
        A_1 = np.maximum(0, Z_1)

        Z_2 = self.w2 @ A_1
        A_2 = 1 / (1 + np.exp(-Z_2))

        return A_1, A_2

    def update_weights_for_one_epoch(self, inputs, outputs, learning_rate):
        x, y_true = inputs, outputs
        A_1, A_2 = self.predict(inputs)
        n = inputs.shape[1]
        
        shared_coefficient = (2 * learning_rate / n) * (y_true - A_2) * A_2 * (1 - A_2)
        relu_gradient = np.where(A_1 > 0, 1, 0)

        dw2 = shared_coefficient @ A_1.T
        dw1 = (self.w2.T @ shared_coefficient * relu_gradient) @ x.T
        self.w1 = self.w1 + dw1
        self.w2 = self.w2 + dw2

    def fit(self, inputs, outputs, learning_rate, epochs=64):
        for i in range(epochs):
            self.update_weights_for_one_epoch(inputs, outputs, learning_rate)

