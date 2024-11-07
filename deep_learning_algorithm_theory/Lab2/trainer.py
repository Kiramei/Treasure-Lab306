import os
import sys

import numpy as np
import torch
from torch import nn
from torch.optim import lr_scheduler
from torchvision import transforms

from model.LeNetPlus import Model



class Logger(object):
    def __init__(self, fileN="Default.log"):
        self.terminal = sys.stdout
        self.log = open(fileN, "w")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.flush()  # 每次写入后刷新到文件中，防止程序意外结束

    def flush(self):
        self.log.flush()


os.environ['KMP_DUPLICATE_LIB_OK']='True'
sys.stdout = Logger("log.txt")
model_name = Model.__dict__['__module__'].split('.')[-1]

data_transform = transforms.Compose([
    transforms.ToTensor()
])

batch_size = 32

data = np.load('util/preprocessed/data_{}.npy'.format(model_name.replace('Plus','')), allow_pickle=True)

cats = data[:12500]
dogs = data[12500:]
np.random.shuffle(cats)
np.random.shuffle(dogs)

cats_train = cats[:int(0.8 * len(cats))]
cats_test = cats[int(0.8 * len(cats)):]

dogs_train = dogs[:int(0.8 * len(dogs))]
dogs_test = dogs[int(0.8 * len(dogs)):]

train_data = np.concatenate([cats_train, dogs_train])
test_data = np.concatenate([cats_test, dogs_test])

train_dataset = torch.utils.data.TensorDataset(torch.tensor(np.array([i[0] for i in train_data]), dtype=torch.float32),
                                               torch.tensor(np.array([i[1] for i in train_data]), dtype=torch.long))
test_dataset = torch.utils.data.TensorDataset(torch.tensor(np.array([i[0] for i in test_data]), dtype=torch.float32),
                                              torch.tensor(np.array([i[1] for i in test_data]), dtype=torch.long))

train_dataloader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

test_dataloader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

model = Model().to(device)

loss_fn = nn.CrossEntropyLoss()

optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.9)

lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

train_losses = []
test_accuracies = []


def train(dataloader, model, loss_fn, optimizer):
    loss, current, n = 0.0, 0.0, 0
    for batch, (x, y) in enumerate(dataloader):
        x, y = x.to(device), y.to(device)
        if x.shape[-1] == 1 or x.shape[-1] == 3:
            x = x.permute(0, 3, 1, 2)
        elif x.shape[1] != 1 or x.shape[1] != 3:
            x = x.unsqueeze(1)
        output = model(x)
        pred = torch.max(output, 1)[1]
        cur_loss = loss_fn(output, y)
        cur_acc = torch.sum(y == pred) / output.shape[0]

        optimizer.zero_grad()
        cur_loss.backward()
        optimizer.step()

        loss += cur_loss.item()
        current += cur_acc.item()
        n = n + 1

    train_loss = loss / n
    train_acc = current / n
    print('train_loss:', str(train_loss))
    print('train_acc:', str(train_acc))
    train_losses.append(train_loss)


def val(dataloader, model, loss_fn):
    model.eval()
    loss, current, n = 0.0, 0.0, 0
    with torch.no_grad():
        for batch, (x, y) in enumerate(dataloader):
            x, y = x.to(device), y.to(device)
            if x.shape[-1] == 1 or x.shape[-1] == 3:
                x = x.permute(0, 3, 1, 2)
            elif x.shape[1] != 1 or x.shape[1] != 3:
                x = x.unsqueeze(1)
            output = model(x)
            pred = torch.max(output, 1)[1]
            cur_loss = loss_fn(output, y)
            cur_acc = torch.sum(y == pred) / output.shape[0]
            loss += cur_loss.item()
            current += cur_acc.item()
            n = n + 1
        print("val_loss：" + str(loss / n))
        print("val_acc：" + str(current / n))
        test_accuracies.append(current / n)
        return current / n


epoch = 100
min_acc = 0
for t in range(epoch):
    print(f'epoch {t + 1}\n---------------')
    train(train_dataloader, model, loss_fn, optimizer)
    a = val(test_dataloader, model, loss_fn)
    if a > min_acc:
        folder = 'checkpoints'
        if not os.path.exists(folder):
            os.mkdir('checkpoints')
        min_acc = a
        print('Save best model')
        torch.save(model.state_dict(),
                   'checkpoints/best_model_{}.pth'.format(model_name))

# write to file
if not os.path.exists('results'): os.mkdir('results')

with open('./results/train_losses_{}.txt'.format(model_name), 'w') as f:
    for item in train_losses:
        f.write("%s\n" % item)

with open('./results/test_accuracies_{}.txt'.format(model_name), 'w') as f:
    for item in test_accuracies:
        f.write("%s\n" % item)
# visualize

import matplotlib.pyplot as plt

plt.plot(train_losses)
plt.title('Train Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.savefig('./results/train_losses_{}.png'.format(model_name))
plt.show()

plt.plot(test_accuracies)
plt.title('Test Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.savefig('./results/test_accuracies_{}.png'.format(model_name))
plt.show()

print('Done')
