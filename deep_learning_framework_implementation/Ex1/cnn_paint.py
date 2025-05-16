import matplotlib.pyplot as plt
import numpy as np

cross = ''
focal = ''

cross_loss =[]
cross_acc = []

focal_loss = []
focal_acc = []

with open('CrossEntropyWithSoftmax.txt', 'r') as f:

    cross = f.read().split('\n')
    cross_loss_ = []
    for cross_unit in cross:
        if cross_unit.count('loss') != 0:
            cross_loss_ .append(float(cross_unit[cross_unit.index('loss: ')+6:]))
        elif cross_unit.count('accuracy') != 0:
            cross_loss_ = np.array(cross_loss_).mean()
            cross_acc.append( float(cross_unit[cross_unit.index('accuracy: ')+10:]))    
            cross_loss.append(cross_loss_)
            cross_loss_ = []
    
with open('FocalLoss.txt', 'r') as f:
    focal = f.read().split('\n')
    focal_loss_ = []
    for focal_unit in focal:
        if focal_unit.count('loss') != 0:
            focal_loss_ .append(float(focal_unit[focal_unit.index('loss: ')+6:]))
        elif focal_unit.count('accuracy') != 0:
            focal_loss_ = np.array(focal_loss_).mean()
            focal_acc.append( float(focal_unit[focal_unit.index('accuracy: ')+10:]))    
            focal_loss.append(focal_loss_)
            focal_loss_ = []
            

plt.plot(cross_loss, label='CrossEntropyWithSoftmax')
plt.plot(focal_loss, label='FocalLoss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

plt.plot(cross_acc, label='CrossEntropyWithSoftmax')
plt.plot(focal_acc, label='FocalLoss')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()    

# print(cross)
# print(focal)