import tensorflow as tf
from sklearn.metrics import f1_score

class CustomModel(tf.Module):
    def __init__(self, input_features):
        initializer = tf.keras.initializers.GlorotNormal()
        self.w1 = tf.Variable(initializer(shape=[input_features, 5000]), name='w1')
        self.bias1 = tf.Variable(tf.zeros([5000]), name='bias1')

        self.w2 = tf.Variable(initializer(shape=[5000, 1000]), name='w2')
        self.bias2 = tf.Variable(tf.zeros(1000), name='bias2')

        self.w3 = tf.Variable(initializer(shape=[1000, 500]), name='w3')
        self.bias3 = tf.Variable(tf.zeros(500), name='bias3')

        self.w_output = tf.Variable(initializer(shape=[500, 1]), name='w_output')
        self.bias_output = tf.Variable(tf.zeros(1), name='bias_output')

    def forward(self, x):
        first = tf.nn.relu(tf.add(tf.matmul(x, self.w1), self.bias1))
        second = tf.nn.relu(tf.add(tf.matmul(first, self.w2), self.bias2))
        third = tf.nn.relu(tf.add(tf.matmul(second, self.w3), self.bias3))
        output = tf.nn.sigmoid(tf.add(tf.matmul(third, self.w_output), self.bias_output))
        return output

    def predict(self, x):
        batched_dataset = tf.data.Dataset.from_tensor_slices(x).batch(len(x))
        X_batch = next(iter(batched_dataset))
        output = self.forward(X_batch)
        return output

    @tf.function
    def fit_one_batch(self, inputs, targets, optimizer, loss_fn):
        with tf.GradientTape() as tape:
            predictions = self.forward(inputs)
            loss = loss_fn(targets, predictions)

        gradients = tape.gradient(loss, self.trainable_variables)
        optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        return loss

    def fit(self, train_dataset, valid_dataset, epochs, BATCH_SIZE,
            optimizer=tf.optimizers.Adam(), loss_fn=tf.losses.BinaryCrossentropy()):
        batched_dataset = train_dataset.batch(BATCH_SIZE)
        batched_validation_dataset = valid_dataset.batch(len(valid_dataset))
        X_valid, y_valid = next(iter(batched_validation_dataset))

        for epoch in range(epochs):
            for X_batch, y_batch in batched_dataset:
                loss = self.fit_one_batch(X_batch, y_batch, optimizer, loss_fn)

            val_predictions = self.predict(X_valid)
            val_loss = loss_fn(y_valid, val_predictions)

            val_f1 = f1_score(y_valid.numpy(), tf.round(val_predictions).numpy(), average='weighted')

            print(f'Epoch {epoch+1}, Loss: {loss.numpy()}, Validation Loss: {val_loss.numpy()}, Validation F1 score: {val_f1}')
