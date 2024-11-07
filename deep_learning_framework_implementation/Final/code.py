import os
import time
import threading
import numpy as np
import multiprocessing
import matrixslow as ms
from functools import partial
import matplotlib.pyplot as plt


class MeanAbsoluteErrorLoss(ms.ops.LossFunction):
    def compute(self):
        assert len(self.parents) == 2
        y = self.parents[0].value
        y_wish = self.parents[1].value  # target ground truth
        self.value = np.mat(1 / len(y) * np.sum(np.abs(y - y_wish)))

    def get_jacobi(self, parent):
        y = self.parents[0].value
        y_wish = self.parents[1].value
        sign = (1 if parent is self.parents[0] else -1)
        return (1 / len(y) * np.sign((y - y_wish) * sign)).T


class Tanh(ms.ops.Operator):
    def compute(self):
        x = self.parents[0].value
        self.value = np.tanh(x)

    def get_jacobi(self, parent):
        return np.diag((1 - np.square(np.tanh(self.parents[0].value))).A1)


def fc(_input, input_size, output_size, activation, graph):
    weights = ms.core.Variable((output_size, input_size), init=True, trainable=True, graph=graph)
    bias = ms.core.Variable((output_size, 1), init=True, trainable=True, graph=graph)
    affine = ms.ops.Add(ms.ops.MatMul(weights, _input, graph=graph), bias, graph=graph)
    return ms.ops.ReLU(affine, graph=graph) if activation == "ReLU" else affine


class Trainer:

    def __init__(self, writer_lock, plot_lock, **kwargs):
        assert plot_lock is not None, "Plot Lock is required!"
        assert writer_lock is not None, "Writer Lock is required!"
        # Initialization-required Parameters
        self.num_features = 0
        self.accuracy = None
        self.test_data = None
        self.train_data = None
        self.time_elapsed = None
        self.train_loss, self.test_loss = [], []
        self.test_predictions, self.test_labels = None, None
        # Hyperparameters
        self.data_dir = kwargs.get("data_dir", "data/Daily_ZX.csv")
        self.state_dim = kwargs.get("state_dim", 12)
        self.sequence_length = kwargs.get("sequence_length", 50)
        self.learning_rate = kwargs.get("learning_rate", 0.005)
        self.batch_size = kwargs.get("batch_size", 16)
        self.num_epochs = kwargs.get("num_epochs", 30)
        self.mlp_layers = kwargs.get("mlp_layers", [40, 10])
        self.train_ratio = kwargs.get("train_ratio", 0.7)
        self.model = kwargs.get("model", "RNN")
        # Hidden Parameters
        self.__ex_name__ = ""
        self.__rev_norm__ = None
        self.__best_loss__ = float("inf")
        self.__plot_lock__ = plot_lock
        self.__writer_lock__ = writer_lock
        self.__graph__ = ms.core.Graph()

    def load_and_normalize_data(self):
        data = np.loadtxt(self.data_dir, delimiter=",", usecols=range(2, 10), dtype=str)[1:].astype(np.float_)
        self.__rev_norm__ = partial(lambda x, mx, mn: x * (mx - mn) + mn, mx=np.max(data), mn=np.min(data))
        data = (data - np.min(data, axis=0)) / (np.max(data, axis=0) - np.min(data, axis=0))
        num_examples, self.num_features = data.shape
        train_size = int(num_examples * self.train_ratio)
        train_idx = (train_size // ((self.sequence_length + 1) * self.num_features)) * (
                (self.sequence_length + 1) * self.num_features)
        test_idx = ((num_examples - train_size) // ((self.sequence_length + 1) * self.num_features)) * (
                (self.sequence_length + 1) * self.num_features)

        self.train_data = data[:train_idx].reshape(-1, self.sequence_length + 1, self.num_features)
        self.test_data = data[num_examples - test_idx:].reshape(-1, self.sequence_length + 1, self.num_features)

    def preprocess_data(self):
        def average_sequence(data, _interval):
            averaged_data = []
            for seq in data:
                averaged_seq = [np.mean(seq[j * _interval:(j + 1) * _interval], axis=0) for j in
                                range(len(seq) // _interval)]
                if len(averaged_seq) * _interval > len(seq):
                    averaged_seq.append(seq[-1])
                averaged_data.append(averaged_seq)
            return np.array(averaged_data)

        interval = self.sequence_length // 5 if self.sequence_length > 5 else 3
        self.train_data = average_sequence(self.train_data, interval)
        self.test_data = average_sequence(self.test_data, interval)

    def build_model(self):
        input_dim = self.num_features
        state_dim = self.state_dim
        inputs = [ms.core.Variable(dim=(input_dim, 1), init=False, trainable=False, graph=self.__graph__) for _ in
                  range(5)]
        if self.model == "RNN":
            U = ms.core.Variable(dim=(state_dim, input_dim), init=True, trainable=True, graph=self.__graph__)
            W = ms.core.Variable(dim=(state_dim, state_dim), init=True, trainable=True, graph=self.__graph__)
            b = ms.core.Variable(dim=(state_dim, 1), init=True, trainable=True, graph=self.__graph__)
            last_state = None
            for input_vec in inputs:
                state = ms.ops.Add(ms.ops.MatMul(U, input_vec, graph=self.__graph__), b, graph=self.__graph__)
                if last_state is not None:
                    state = ms.ops.Add(ms.ops.MatMul(W, last_state, graph=self.__graph__), state, graph=self.__graph__)
                state = ms.ops.ReLU(state, graph=self.__graph__)
                last_state = state
            fc1 = fc(last_state, state_dim, 40, "ReLU", graph=self.__graph__)
            fc2 = fc(fc1, 40, 10, "ReLU", graph=self.__graph__)
            output = fc(fc2, 10, 1, "None", graph=self.__graph__)
            prediction = ms.ops.ReLU(output, graph=self.__graph__)
        elif self.model == "LSTM":
            def create_lstm_weights():
                return {
                    "U": ms.core.Variable(dim=(state_dim, input_dim), init=True, trainable=True, graph=self.__graph__),
                    "W": ms.core.Variable(dim=(state_dim, state_dim), init=True, trainable=True, graph=self.__graph__),
                    "b": ms.core.Variable(dim=(state_dim, 1), init=True, trainable=True, graph=self.__graph__)
                }

            weights = {
                "forget": create_lstm_weights(),
                "input": create_lstm_weights(),
                "candidate": create_lstm_weights(),
                "output": create_lstm_weights()
            }
            last_hidden = None
            cell_state = None
            for input_vec in inputs:
                forget_gate = ms.ops.Add(ms.ops.MatMul(weights["forget"]["U"], input_vec, graph=self.__graph__),
                                         weights["forget"]["b"], graph=self.__graph__)
                input_gate = ms.ops.Add(ms.ops.MatMul(weights["input"]["U"], input_vec, graph=self.__graph__),
                                        weights["input"]["b"], graph=self.__graph__)
                candidate_cell = ms.ops.Add(ms.ops.MatMul(weights["candidate"]["U"], input_vec, graph=self.__graph__),
                                            weights["candidate"]["b"], graph=self.__graph__)
                output_gate = ms.ops.Add(ms.ops.MatMul(weights["output"]["U"], input_vec, graph=self.__graph__),
                                         weights["output"]["b"], graph=self.__graph__)
                if last_hidden is not None:
                    forget_gate = ms.ops.Add(ms.ops.MatMul(weights["forget"]["W"], last_hidden, graph=self.__graph__),
                                             forget_gate, graph=self.__graph__)
                    input_gate = ms.ops.Add(ms.ops.MatMul(weights["input"]["W"], last_hidden, graph=self.__graph__),
                                            input_gate, graph=self.__graph__)
                    candidate_cell = ms.ops.Add(
                        ms.ops.MatMul(weights["candidate"]["W"], last_hidden, graph=self.__graph__),
                        candidate_cell, graph=self.__graph__)
                    output_gate = ms.ops.Add(ms.ops.MatMul(weights["output"]["W"], last_hidden, graph=self.__graph__),
                                             output_gate, graph=self.__graph__)
                forget_gate = ms.ops.Logistic(forget_gate, graph=self.__graph__)
                input_gate = ms.ops.Logistic(input_gate, graph=self.__graph__)
                candidate_cell = Tanh(candidate_cell, graph=self.__graph__)
                if cell_state is not None:
                    cell_state = ms.ops.Add(ms.ops.Multiply(forget_gate, cell_state, graph=self.__graph__),
                                            ms.ops.Multiply(input_gate, candidate_cell, graph=self.__graph__),
                                            graph=self.__graph__)
                else:
                    cell_state = ms.ops.Multiply(input_gate, candidate_cell, graph=self.__graph__)
                last_hidden = ms.ops.Multiply(output_gate, Tanh(cell_state, graph=self.__graph__), graph=self.__graph__)
            fc1 = fc(last_hidden, state_dim, 40, "ReLU", graph=self.__graph__)
            fc2 = fc(fc1, 40, 10, "ReLU", graph=self.__graph__)
            output = fc(fc2, 10, 1, "None", graph=self.__graph__)
            prediction = ms.ops.ReLU(output, graph=self.__graph__)
        elif self.model == "MLP":
            fc1 = fc(inputs[-1], input_dim, self.mlp_layers[0], "ReLU", graph=self.__graph__)
            fc2 = fc(fc1, self.mlp_layers[0], self.mlp_layers[1], "ReLU", graph=self.__graph__)
            output = fc(fc2, self.mlp_layers[1], 1, "None", graph=self.__graph__)
            prediction = ms.ops.ReLU(output, graph=self.__graph__)
        else:
            raise ValueError("Invalid Model Type!")
        return inputs, prediction

    def train_and_evaluate(self, input_nodes, pred_node):
        label = ms.core.Variable((1, 1), trainable=False, graph=self.__graph__)
        loss = MeanAbsoluteErrorLoss(pred_node, label, graph=self.__graph__)
        optimizer = ms.optimizer.Adam(self.__graph__, loss, self.learning_rate)

        for epoch in range(self.num_epochs):
            batch_loss = 0
            for batch_idx, sequence in enumerate(self.train_data):
                for step, input_var in enumerate(input_nodes):
                    input_var.set_value(np.mat(sequence[step]).T)
                label.set_value(np.mat(sequence[-1, 2]).T)
                optimizer.one_step()
                if (batch_idx + 1) % self.batch_size == 0:
                    print(f"Epoch: {epoch + 1}, Iteration: {batch_idx + 1}, Loss: {batch_loss:.3f}")
                    batch_loss = loss.value[0, 0]
                    optimizer.update()
            self.train_loss.append(batch_loss)

            test_predictions, test_labels = [], []
            for sequence in self.test_data:
                for step, input_var in enumerate(input_nodes):
                    input_var.set_value(np.mat(sequence[step]).T)
                pred_node.forward()
                test_predictions.append(pred_node.value.A.ravel())
                test_labels.append(sequence[-1, 2].T)
            test_predictions = np.array(test_predictions)
            test_labels = np.array(test_labels).reshape(len(self.test_data), 1)
            loss_mae = np.mean(np.abs(test_predictions - test_labels))
            accuracy = np.mean(1 - np.abs(test_predictions - test_labels) / test_labels) * 100
            self.test_loss.append(loss_mae)
            if loss_mae < self.__best_loss__:
                self.__best_loss__ = loss_mae
                self.accuracy = accuracy
                self.test_labels = test_labels
                self.test_predictions = test_predictions

            print(f"Model: {self.__ex_name__}, Epoch: {epoch + 1}, Test Loss: {loss_mae:.5f}")

    def plot_result(self):
        vis_test_predictions = self.__rev_norm__(self.test_predictions.flatten())
        vis_test_labels = self.__rev_norm__(self.test_labels.flatten())

        self.__plot_lock__.acquire()
        plt.cla()
        plt.xlabel("Epoch")
        plt.ylabel("MAE")
        plt.title(f'Model: {self.model}  History: {self.sequence_length} days')
        plt.plot(np.arange(len(self.train_loss)), self.train_loss,
                 np.arange(len(self.test_loss)), self.test_loss)
        plt.legend(['Train Loss', 'Test Loss'])
        plt.savefig(f"results/{self.__ex_name__}/Loss.png")

        plt.cla()
        plt.xlabel("Day")
        plt.ylabel("Open")
        plt.title(f'Model: {self.model}  History: {self.sequence_length} days')
        plt.plot(np.arange(len(vis_test_predictions)), vis_test_predictions,
                 np.arange(len(vis_test_labels)), vis_test_labels)
        plt.legend(['Predicted', 'True'])
        plt.savefig(f"results/{self.__ex_name__}/Test.png")
        self.__plot_lock__.release()

    def write_result(self):
        self.__best_loss__ = self.__rev_norm__(self.__best_loss__)
        self.__writer_lock__.acquire()
        if not os.path.exists("results/results.csv"):
            with open("results/results.csv", "w") as f:
                f.write("Model,Sequence Length,Best Test Loss,Accuracy,Time Elapsed\n")
        with open("results/results.csv", "a") as f:
            f.write(
                f"{self.__ex_name__},{self.sequence_length},{self.__best_loss__},{self.accuracy},{self.time_elapsed}\n")
        self.__writer_lock__.release()

    def execute(self):
        self.__ex_name__ = f"{self.model}_seq{self.sequence_length}"
        if not os.path.exists("results"):
            os.makedirs("results")
        if not os.path.exists(f"results/{self.__ex_name__}"):
            os.makedirs(f"results/{self.__ex_name__}")
        self.load_and_normalize_data()
        if self.sequence_length > 5:
            self.preprocess_data()
        start_time = time.time()
        inputs, prediction = self.build_model()
        self.train_and_evaluate(inputs, prediction)
        self.time_elapsed = time.time() - start_time
        self.plot_result()
        self.write_result()
        print(f"Model: {self.__ex_name__} Best Test Loss: {self.__best_loss__}, Time Taken: {self.time_elapsed:.2f}s")

    def execute_async(self):
        threading.Thread(target=self.execute).start()


wt_lock = threading.Lock()
plt_lock = threading.Lock()


def unit_time(_seq_len):
    model_selected = ['RNN', 'LSTM', 'MLP']
    for model in model_selected:
        Trainer(wt_lock, plt_lock, model=model, sequence_length=_seq_len, num_epochs=50).execute_async()


if __name__ == "__main__":
    time_selected = [15, 20, 25, 30, 35, 40, 45, 50]
    for seq_len in time_selected:
        multiprocessing.Process(target=unit_time, args=[seq_len]).start()
