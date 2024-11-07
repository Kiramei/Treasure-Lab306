# ==================== Importing Libraries ====================

import os

import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# 输出重定向
import sys


def print(*args, **kwargs):
    kwargs['file'] = sys.stdout
    __builtins__.print(*args, **kwargs)


# ================== End of Importing Libraries ==================

batch_size = 1000
train_epochs = 10000
hidden_size = 1024


# define a Dataloader function
def get_heartbeat_dataloader(train_root, test_root, bs=100, val_split_factor=0.2):
    train_df = pd.read_csv(train_root, header=None)
    test_df = pd.read_csv(test_root, header=None)

    train_data = train_df.to_numpy()
    test_data = test_df.to_numpy()

    np.random.shuffle(train_data)
    first_class = train_data[train_data[:, -1] == 0]
    second_class = train_data[train_data[:, -1] == 1]
    third_class = train_data[train_data[:, -1] == 2]
    fourth_class = train_data[train_data[:, -1] == 3]
    fifth_class = train_data[train_data[:, -1] == 4]

    DATA_LENGTH = min(len(first_class), len(second_class), len(third_class), len(fourth_class), len(fifth_class))
    train_len = int(DATA_LENGTH * (1 - val_split_factor))

    train_data = np.concatenate((first_class[:train_len], second_class[:train_len], third_class[:train_len],
                                 fourth_class[:train_len], fifth_class[:train_len]))
    val_data = np.concatenate((first_class[train_len:], second_class[train_len:], third_class[train_len:],
                               fourth_class[train_len:], fifth_class[train_len:]))

    train_dataset = torch.utils.data.TensorDataset(torch.from_numpy(train_data[:, :-1]).float(),
                                                   torch.from_numpy(train_data[:, -1]).long(), )
    val_dataset = torch.utils.data.TensorDataset(torch.from_numpy(val_data[:, :-1]).float(),
                                                 torch.from_numpy(val_data[:, -1]).long())
    test_dataset = torch.utils.data.TensorDataset(torch.from_numpy(test_data[:, :-1]).float(),
                                                  torch.from_numpy(test_data[:, -1]).long())

    train_loader = DataLoader(train_dataset, batch_size=bs, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=bs, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=bs, shuffle=True)

    return train_loader, val_loader, test_loader


# =============== Declare Dataloader ====================
print('================ Start Loading Data ================')
train_loader, val_loader, test_loader = get_heartbeat_dataloader('./data/mitbih_train.csv',
                                                                 './data/mitbih_test.csv',
                                                                 bs=batch_size,
                                                                 val_split_factor=0.2)
print('Train Data Length: ', len(train_loader.dataset))
print('Validation Data Length: ', len(val_loader.dataset))
print('Test Data Length: ', len(test_loader.dataset))
print('================ End Loading Data ==================')

# ================ End of Dataloader ====================


# ==================== Start Model Field ====================

"""
Model One: Attention Based Classifier: Optimized Attention Model
Introduction: This model is a simple self-attention based classifier. It uses a self-attention mechanism to
                capture the dependencies between different time steps in the input sequence.

                The input is first passed through a linear layer to embed the input features to a higher dimension. 
                The embedded input is then passed through three linear layers to generate the query, key, and value
                matrices for the self-attention mechanism. The attention scores are computed using the query and key
                matrices, and the attention weights are computed using the softmax function. The weighted sum of the
                values is then passed through a fully connected layer to generate the output.

                The model uses dropout for regularization and has a fully connected layer at the end to generate the
                output logits.

Structure: The model consists of the following layers:
            1. Embedding layer: Linear layer to embed the input features to a higher dimension.
            2. Query linear layer: Linear layer to generate the query matrix for self-attention.
            3. Key linear layer: Linear layer to generate the key matrix for self-attention.
            4. Value linear layer: Linear layer to generate the value matrix for self-attention.
            5. Dropout layer: Dropout layer for regularization.
            6. MLP: Multi-layer perceptron to generate the output logits.

Parameters:
    - input_size: The input feature dimension.
    - hidden_size: The hidden size for the linear layers.
    - num_classes: The number of output classes.
    - dropout: The dropout probability.

Forward Function: The forward function takes the input tensor x of shape (batch_size, seq_len, input_size) as input
                    and returns the output tensor of shape (batch_size, num_classes).

"""


class SelfAttentionClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes, dropout=0.2):
        super(SelfAttentionClassifier, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_classes = num_classes

        self.embedding = nn.Linear(input_size, hidden_size)
        self.q_linear = nn.Linear(hidden_size, hidden_size)
        self.k_linear = nn.Linear(hidden_size, hidden_size)
        self.v_linear = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, num_classes)
        self.activation = nn.ReLU()
        self.classification = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_classes)
        )

    def forward(self, x):
        x = x.unsqueeze(2).transpose(1, 2)
        B, seq_len, _ = x.shape
        # Embed the input
        embedded = self.embedding(x)
        # Apply self-attention
        Q = self.q_linear(embedded)
        K = self.k_linear(embedded)
        V = self.v_linear(embedded)
        # Compute attention scores
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.hidden_size ** 0.5)
        attention_weights = F.softmax(attention_scores, dim=-1)
        # Compute weighted sum of values
        attention_output = torch.matmul(attention_weights, V)
        # Apply dropout and pass through the fully connected layer
        output = self.dropout(attention_output.mean(dim=1))
        output = self.classification(output)
        return output


# ============================================================================================

"""
Model Two: LSTM Classifier
Introduction: This model is a simple LSTM classifier. It uses an LSTM layer to capture the temporal dependencies
                in the input sequence.
                
                The input is passed through an LSTM layer with a hidden size of 128. The output of the LSTM layer
                is passed through a fully connected layer to generate the output logits.
                
Structure: The model consists of the following layers:
            1. LSTM layer: LSTM layer with a hidden size of 128. Its detailed implementation is as follows:
                - Initialize the hidden state and cell state to zeros.
                - Iterate over the input sequence and update the hidden state and cell state at each time step.
                - Use the hidden state at the last time step as the output of the LSTM layer.
            
            2. Fully connected layer: Linear layer to generate the output logits.
            
Parameters:
    - input_size: The input feature dimension.
    - hidden_size: The hidden size for the LSTM layer.
    - num_layers: The number of LSTM layers.
    - num_classes: The number of output classes.
    
Forward Function: The forward function takes the input tensor x of shape (batch_size, seq_len, input_size) as input
                    and returns the output tensor of shape (batch_size, num_classes).
"""


class LSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(LSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        # 定义 LSTM 单元的权重和偏置
        self.W_ih = nn.Parameter(torch.Tensor(4 * hidden_size, input_size))
        self.W_hh = nn.Parameter(torch.Tensor(4 * hidden_size, hidden_size))
        self.bias_ih = nn.Parameter(torch.Tensor(4 * hidden_size))
        self.bias_hh = nn.Parameter(torch.Tensor(4 * hidden_size))

        # 初始化权重和偏置
        self.reset_parameters()

    def reset_parameters(self):
        # 使用Xavier初始化权重
        nn.init.xavier_uniform_(self.W_ih)
        nn.init.xavier_uniform_(self.W_hh)
        # 将偏置初始化为0
        nn.init.zeros_(self.bias_ih)
        nn.init.zeros_(self.bias_hh)

    def forward(self, input, states):
        h_prev, c_prev = states

        # 计算 LSTM 单元的四个门
        with torch.no_grad():
            gates = F.linear(input, self.W_ih, self.bias_ih) + \
                    F.linear(h_prev, self.W_hh, self.bias_hh)

        # 拆分为四个门
        i, f, g, o = gates.chunk(4, 1)

        # 计算新的细胞状态和隐藏状态
        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        g = torch.tanh(g)
        o = torch.sigmoid(o)
        # c_next = f * c_prev + i * g
        with torch.no_grad():
            c_next = f * c_prev + i * g
        # c_next = torch.add(f * c_prev, i * g)
        h_next = o * torch.tanh(c_next)

        return h_next, c_next


class LSTMClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, dropout=0.0):
        super(LSTMClassifier, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # 创建多层 LSTM 单元
        self.lstm_cells = nn.ModuleList([LSTMCell(input_size if i == 0 else hidden_size, hidden_size)
                                         for i in range(num_layers)])

        # 如果需要,添加dropout层
        if dropout > 0:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = None

        self.classification = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_classes)
        )

    def forward(self, x, initial_states=None):
        x = x.unsqueeze(2).transpose(1, 2)
        batch_size, seq_len, _ = x.size()

        # 如果没有提供初始状态,则初始化为0
        if initial_states is None:
            h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=x.device)
            c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=x.device)
        else:
            h0, c0 = initial_states

        # 遍历时间步长
        outputs = []
        h, c = h0, c0
        for t in range(seq_len):
            for i in range(self.num_layers):
                h[i], c[i] = self.lstm_cells[i](x[:, t, :], (h[i], c[i]))
            if self.dropout is not None:
                h = self.dropout(h)
            outputs.append(h[-1])

        # 返回输出序列和最终状态
        output = torch.stack(outputs, dim=1)[:, -1]
        output = self.classification(output)
        return output


"""
Model Three: Multi-head Attention Based Classifier: Optimized Attention Model
Introduction: This model is a simple self-attention based classifier. It uses a self-attention mechanism to
                capture the dependencies between different time steps in the input sequence.

                The input is first passed through a linear layer to embed the input features to a higher dimension. 
                The embedded input is then passed through three linear layers to generate the query, key, and value
                matrices for the self-attention mechanism. The attention scores are computed using the query and key
                matrices, and the attention weights are computed using the softmax function. The weighted sum of the
                values is then passed through a fully connected layer to generate the output.

                The model uses dropout for regularization and has a fully connected layer at the end to generate the
                output logits.

Structure: The model consists of the following layers:
            1. Embedding layer: Linear layer to embed the input features to a higher dimension.
            2. Query linear layer: Linear layer to generate the query matrix for self-attention.
            3. Key linear layer: Linear layer to generate the key matrix for self-attention.
            4. Value linear layer: Linear layer to generate the value matrix for self-attention.
            5. Dropout layer: Dropout layer for regularization.
            6. Multi-head self-attention: Self-attention mechanism with multiple heads.
            7. MLP: Multi-layer perceptron to generate the output logits.

Parameters:
    - input_size: The input feature dimension.
    - hidden_size: The hidden size for the linear layers.
    - num_classes: The number of output classes.
    - num_heads: The number of attention heads.
    - dropout: The dropout probability.

Forward Function: The forward function takes the input tensor x of shape (batch_size, seq_len, input_size) as input
                    and returns the output tensor of shape (batch_size, num_classes).

"""


class MultiHeadSelfAttentionClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes, num_heads, dropout=0.2):
        super(MultiHeadSelfAttentionClassifier, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.num_heads = num_heads

        self.embedding = nn.Linear(input_size, hidden_size)
        self.q_linears = nn.ModuleList([nn.Linear(hidden_size, hidden_size) for _ in range(num_heads)])
        self.k_linears = nn.ModuleList([nn.Linear(hidden_size, hidden_size) for _ in range(num_heads)])
        self.v_linears = nn.ModuleList([nn.Linear(hidden_size, hidden_size) for _ in range(num_heads)])
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size * num_heads, num_classes)
        self.activation = nn.ReLU()
        self.classification = nn.Sequential(
            nn.Linear(hidden_size * num_heads, hidden_size * num_heads),
            nn.ReLU(),
            nn.Linear(hidden_size * num_heads, num_classes)
        )

    def forward(self, x):
        x = x.unsqueeze(2).transpose(1, 2)
        B, seq_len, _ = x.shape

        # Embed the input
        embedded = self.embedding(x)

        # Apply self-attention
        attention_outputs = []
        for i in range(self.num_heads):
            Q = self.q_linears[i](embedded)
            K = self.k_linears[i](embedded)
            V = self.v_linears[i](embedded)

            # Compute attention scores
            attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.hidden_size ** 0.5)
            attention_weights = F.softmax(attention_scores, dim=-1)

            # Compute weighted sum of values
            attention_output = torch.matmul(attention_weights, V)
            attention_outputs.append(attention_output)

        # Concatenate the outputs of all heads
        attention_output = torch.cat(attention_outputs, dim=-1)

        # Apply dropout and pass through the fully connected layer
        output = self.dropout(attention_output.mean(dim=1))
        output = self.classification(output)

        return output


# ==================== End Model Field ====================


# ==================== Start Training Field ====================
def train(model, train_loader, val_loader, num_epochs=10, learning_rate=0.01):
    loss_list = []
    accuracy_list = []
    best_accuracy = 0
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    total_step = len(train_loader)
    for epoch in range(num_epochs):
        model.train()
        loss_all = 0
        for i, (sequences, labels) in enumerate(train_loader):
            sequences = sequences.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(sequences)
            loss = criterion(outputs, labels)
            loss_all += loss.item()
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i + 1) % 100 == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                      .format(epoch + 1, num_epochs, i + 1, total_step, loss.item()))
        mean_loss = loss_all / len(train_loader)
        loss_list.append(mean_loss)

        # Validate the model
        model.eval()
        with torch.no_grad():
            correct = 0
            total = 0
            for sequences, labels in val_loader:
                sequences = sequences.to(device)
                labels = labels.to(device)
                outputs = model(sequences)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            accuracy = 100 * correct / total
            accuracy_list.append(accuracy)
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                torch.save(model.state_dict(), 'best_model_{}.ckpt'.format(model.__class__.__name__))
            print('Epoch {} > Validation Accuracy: {} %'.format(epoch, round(accuracy, 2)))

    return model, loss_list, accuracy_list, best_accuracy


# ==================== End Training Field ====================

# ==================== Start Testing Field ====================
def test(model, test_loader):
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        predicted_list = []
        gt_list = []
        for sequences, labels in test_loader:
            sequences = sequences.to(device)
            labels = labels.to(device)
            outputs = model(sequences)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            predicted = predicted.cpu().numpy()
            labels = labels.cpu().numpy()
            predicted_list.append(predicted)
            gt_list.append(labels)
            correct += (predicted == labels).sum().item()
        print('Test Accuracy of the model on the 10000 test sequences: {} %'.format(100 * correct / total))
        predicted_list = np.concatenate(predicted_list)
        gt_list = np.concatenate(gt_list)
        return predicted_list, gt_list


# ==================== End Testing Field ====================

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def start_model_one():
    # ==================== Start Training and Testing ====================
    # Model One: Self-Attention Classifier
    model1 = SelfAttentionClassifier(input_size=187, hidden_size=hidden_size, num_classes=5, dropout=0.2).to(device)
    print('================ Start Training Model One: Self-Attention Classifier ================')
    model1, model1_loss_list, model1_accuracy_list, model1_best_accuracy = \
        train(model1, train_loader, val_loader, num_epochs=train_epochs, learning_rate=0.001)
    print('================ End Training Model One: Self-Attention Classifier ==================')
    print('================ Start Testing Model One: Self-Attention Classifier ================')
    predicted_list_model1, gt_list_model1 = test(model1, test_loader)
    print('================ End Testing Model One: Self-Attention Classifier ==================')
    torch.save(model1.state_dict(), 'model1.ckpt')
    # ==================== Start Result Visualization ====================
    # Model One: Self-Attention Classifier
    # Curve Visualization
    plt.figure(figsize=(10, 5))
    # Loss Curve
    plt.subplot(2, 1, 1)
    plt.plot(model1_loss_list)
    plt.title("Model One: Self-Attention Classifier Loss Curve")
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    # Accuracy Curve
    plt.subplot(2, 1, 2)
    plt.plot(model1_accuracy_list)
    plt.title("Model One: Self-Attention Classifier Accuracy Curve")
    plt.xlabel("Iterations")
    plt.ylabel("Accuracy")
    # Show
    plt.tight_layout()
    plt.show()
    # Draw Confusion Matrix
    conf_matrix = confusion_matrix(gt_list_model1, predicted_list_model1)
    plt.figure(figsize=(10, 5))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
    plt.title("Model One: Self-Attention Classifier Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.show()
    # ==================== End Result Visualization ====================

    # ==================== Save Result  ====================
    np.save('model1_predicted.npy', predicted_list_model1)
    np.save('model1_gt.npy', gt_list_model1)
    np.save('model1_loss.npy', model1_loss_list)
    np.save('model1_accuracy.npy', model1_accuracy_list)
    # ==================== End Save Result  ====================


def start_model_two():
    # Model Two: LSTM Classifier
    model2 = LSTMClassifier(input_size=187, hidden_size=hidden_size, num_layers=1, num_classes=5).to(device)
    print('================ Start Training Model Two: LSTM Classifier ================')
    model2, model2_loss_list, model2_accuracy_list, model2_best_accuracy = \
        train(model2, train_loader, val_loader, num_epochs=train_epochs, learning_rate=0.001)
    print('================ End Training Model Two: LSTM Classifier ==================')
    print('================ Start Testing Model Two: LSTM Classifier ================')
    predicted_list_model2, gt_list_model2 = test(model2, test_loader)
    print('================ End Testing Model Two: LSTM Classifier ==================')
    torch.save(model2.state_dict(), 'model2.ckpt')

    # ==================== End Training and Testing ====================

    # Model Two: LSTM Classifier

    # Curve Visualization
    plt.figure(figsize=(10, 5))
    # Loss Curve
    plt.subplot(2, 1, 1)
    plt.plot(model2_loss_list)
    plt.title("Model Two: LSTM Classifier Loss Curve")
    plt.xlabel("Iterations")
    plt.ylabel("Loss")

    # Accuracy Curve
    plt.subplot(2, 1, 2)
    plt.plot(model2_accuracy_list)
    plt.title("Model Two: LSTM Classifier Accuracy Curve")
    plt.xlabel("Iterations")
    plt.ylabel("Accuracy")

    # Show
    plt.tight_layout()
    plt.show()

    # Draw Confusion Matrix
    conf_matrix = confusion_matrix(gt_list_model2, predicted_list_model2)
    plt.figure(figsize=(10, 5))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
    plt.title("Model Two: LSTM Classifier Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.show()
    # ==================== End Result Visualization ====================

    # ==================== Save Result  ====================
    np.save('model2_loss.npy', model2_loss_list)
    np.save('model2_accuracy.npy', model2_accuracy_list)
    np.save('model2_predicted.npy', predicted_list_model2)
    np.save('model2_gt.npy', gt_list_model2)
    # ==================== End Save Result  ====================


def start_model_three():
    # ==================== Start Training and Testing ====================
    # Model One: Self-Attention Classifier
    model1 = MultiHeadSelfAttentionClassifier(input_size=187, hidden_size=hidden_size, num_classes=5, num_heads=4,
                                              dropout=0.2).to(device)
    print('================ Start Training Model Three: Multi-head Self-Attention Classifier ================')
    model1, model3_loss_list, model3_accuracy_list, model1_best_accuracy = \
        train(model1, train_loader, val_loader, num_epochs=train_epochs, learning_rate=0.001)
    print('================ End Training Model Three: Multi-head Self-Attention Classifier ==================')
    print('================ Start Testing Model Three: Multi-head Self-Attention Classifier ================')
    predicted_list_model3, gt_list_model3 = test(model1, test_loader)
    print('================ End Testing Model Three: Multi-head Self-Attention Classifier ==================')
    torch.save(model1.state_dict(), 'model3.ckpt')
    # ==================== Start Result Visualization ====================
    # Model One: Self-Attention Classifier
    # Curve Visualization
    plt.figure(figsize=(10, 5))
    # Loss Curve
    plt.subplot(2, 1, 1)
    plt.plot(model3_loss_list)
    plt.title("Model Three: Multi-head Self-Attention Classifier Loss Curve")
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    # Accuracy Curve
    plt.subplot(2, 1, 2)
    plt.plot(model3_accuracy_list)
    plt.title("Model Three: Multi-head Self-Attention Classifier Accuracy Curve")
    plt.xlabel("Iterations")
    plt.ylabel("Accuracy")
    # Show
    plt.tight_layout()
    plt.show()
    # Draw Confusion Matrix
    conf_matrix = confusion_matrix(gt_list_model3, predicted_list_model3)
    plt.figure(figsize=(10, 5))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
    plt.title("Model Three: Multi-head Self-Attention Classifier Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.show()
    # ==================== End Result Visualization ====================

    # ==================== Save Result  ====================
    np.save('model3_predicted.npy', predicted_list_model3)
    np.save('model3_gt.npy', gt_list_model3)
    np.save('model3_loss.npy', model3_loss_list)
    np.save('model3_accuracy.npy', model3_accuracy_list)

    # ==================== End Save Result  ====================


start_model_one()
# start_model_two()
# start_model_three()
