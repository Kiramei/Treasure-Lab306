from scipy.special import xlogy
import matrixslow as ms
import numpy as np

from ..core import Node
from ..ops import SoftMax


class LossFunction(Node):
    '''
    定义损失函数抽象类
    '''
    pass


class LogLoss(LossFunction):

    def compute(self):
        assert len(self.parents) == 1

        x = self.parents[0].value

        self.value = np.log(1 + np.power(np.e, np.where(-x > 1e2, 1e2, -x)))

    def get_jacobi(self, parent):
        x = parent.value
        diag = -1 / (1 + np.power(np.e, np.where(x > 1e2, 1e2, x)))

        return np.diag(diag.ravel())


class CrossEntropyWithSoftMax(LossFunction):
    """
    对第一个父节点施加SoftMax之后，再以第二个父节点为标签One-Hot向量计算交叉熵
    """

    def compute(self):
        prob = SoftMax.softmax(self.parents[0].value)
        self.value = np.mat(
            -np.sum(np.multiply(self.parents[1].value, np.log(prob + 1e-10))))

    def get_jacobi(self, parent):
        # 这里存在重复计算，但为了代码清晰简洁，舍弃进一步优化
        prob = SoftMax.softmax(self.parents[0].value)
        if parent is self.parents[0]:
            return (prob - self.parents[1].value).T
        else:
            return (-np.log(prob)).T


class PerceptionLoss(LossFunction):
    """
    感知机损失，输入为正时为0，输入为负时为输入的相反数
    """

    def compute(self):
        self.value = np.mat(np.where(
            self.parents[0].value >= 0.0, 0.0, -self.parents[0].value))

    def get_jacobi(self, parent):
        """
        雅克比矩阵为对角阵，每个对角线元素对应一个父节点元素。若父节点元素大于0，则
        相应对角线元素（偏导数）为0，否则为-1。
        """
        diag = np.where(parent.value >= 0.0, 0.0, -1)
        return np.diag(diag.ravel())


class MeanSquaredErrorLoss(LossFunction):
    def compute(self):
        assert len(self.parents) == 2
        y = self.parents[0].value
        y_wish = self.parents[1].value  # target ground truth
        N = len(y)
        self.value = np.mat(1 / N * np.sum(np.multiply(y - y_wish, y - y_wish)))

    def get_jacobi(self, parent):
        y = self.parents[0].value
        y_wish = self.parents[1].value
        N = len(y)
        if parent is self.parents[0]:
            return (1 / N * 2 * (y - y_wish)).T
        else:
            return (1 / N * 2 * (y_wish - y)).T


class L2NormLoss(LossFunction):
    """
    L2范数损失
    """

    def compute(self):
        assert len(self.parents) == 2
        s1 = self.parents[0].value
        lmd = self.parents[1].value
        self.value = np.mat(lmd * np.sum(np.multiply(s1, s1)))

    def get_jacobi(self, parent):
        s1 = self.parents[0].value
        lmd = self.parents[1].value
        if parent is self.parents[0]:
            return 2 * lmd * s1.T
        else:
            return 0


# class FocalLoss(LossFunction):
#     def compute(self):
#         assert len(self.parents) == 4
#         y_true = self.parents[0].value
#         y_pred = self.parents[1].value
#         GAMMA = self.parents[2].value
#         ALPHA = self.parents[3].value
#
#         pt_1 = np.where(y_true == 1, y_pred, np.ones_like(y_pred))
#         pt_0 = np.where(y_true == 0, y_pred, np.zeros_like(y_pred))
#
#         epsilon = np.finfo(float).eps
#
#         loss_1 = -ALPHA * np.power(1. - pt_1, GAMMA) * xlogy(pt_1, pt_1 + epsilon)
#         loss_0 = -(1 - ALPHA) * np.power(pt_0, GAMMA) * xlogy(1. - pt_0, 1. - pt_0 + epsilon)
#
#         self.value = np.sum(loss_1) - np.sum(loss_0)
#
#     def get_jacobi(self, parent):
#         y_true = self.parents[0].value
#         y_pred = self.parents[1].value
#         GAMMA = self.parents[2].value
#         ALPHA = self.parents[3].value
#
#         pt_1 = np.where(y_true == 1, y_pred, np.ones_like(y_pred))
#         pt_0 = np.where(y_true == 0, y_pred, np.zeros_like(y_pred))
#
#         epsilon = np.finfo(float).eps
#
#         jacobi = np.zeros_like(y_pred)
#
#         if parent is self.parents[0]:
#             jacobi = -ALPHA * GAMMA * np.power(1. - pt_1, GAMMA - 1) * (1. - pt_1) * np.log(pt_1 + epsilon)
#             jacobi += (1 - ALPHA) * GAMMA * np.power(pt_0, GAMMA - 1) * (1. - pt_0) * np.log(1. - pt_0 + epsilon)
#         elif parent is self.parents[1]:
#             jacobi = -ALPHA * GAMMA * np.power(1. - pt_1, GAMMA - 1) * (1. - pt_1) * (1 + np.log(pt_1 + epsilon))
#             jacobi += (1 - ALPHA) * GAMMA * np.power(pt_0, GAMMA - 1) * (1. - pt_0) * (1 + np.log(1. - pt_0 + epsilon))
#         elif parent is self.parents[2]:
#             jacobi = -ALPHA * np.power(1. - pt_1, GAMMA) * xlogy(pt_1, pt_1 + epsilon) * np.log(1. - pt_1 + epsilon)
#             jacobi += -(1 - ALPHA) * np.power(pt_0, GAMMA) * xlogy(1. - pt_0, 1. - pt_0 + epsilon) * np.log(pt_0 + epsilon)
#         elif parent is self.parents[3]:
#             jacobi = -np.power(1. - pt_1, GAMMA) * xlogy(pt_1, pt_1 + epsilon)
#             jacobi += -np.power(pt_0, GAMMA) * xlogy(1. - pt_0, 1. - pt_0 + epsilon)
#
#         return jacobi


"""
class MultiCEFocalLoss(torch.nn.Module):
    def __init__(self, class_num, gamma=2, alpha=None, reduction='mean'):
        super(MultiCEFocalLoss, self).__init__()
        if alpha is None:
            self.alpha = Variable(torch.ones(class_num, 1))
        else:
            self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.class_num =  class_num

    def forward(self, predict, target):
        pt = F.softmax(predict, dim=1) # softmmax获取预测概率
        class_mask = F.one_hot(target, self.class_num) #获取target的one hot编码
        ids = target.view(-1, 1) 
        alpha = self.alpha[ids.data.view(-1)] # 注意，这里的alpha是给定的一个list(tensor),里面的元素分别是每一个类的权重因子
        probs = (pt * class_mask).sum(1).view(-1, 1) # 利用onehot作为mask，提取对应的pt
        log_p = probs.log()  # 同样，原始ce上增加一个动态权重衰减因子
        loss = -alpha * (torch.pow((1 - probs), self.gamma)) * log_p

        if self.reduction == 'mean':
            loss = loss.mean()
        elif self.reduction == 'sum':
            loss = loss.sum()
        return loss
"""

class FocalLoss(LossFunction):
    """
    对第一个父节点施加SoftMax之后，再以第二个父节点为标签One-Hot向量计算交叉熵
    """
    def __init__(self, *parents, gamma=2, alpha=None, class_num=10, reduction='mean'):
        super(FocalLoss, self).__init__(*parents)
        self.gamma = gamma
        if alpha is None:
            self.alpha = ms.Variable(np.ones((class_num, 1)))
        else:
            self.alpha = alpha
        self.class_num = class_num
        self.reduction = reduction


    def compute(self):
        prob = SoftMax.softmax(self.parents[0].value)
        class_mask = np.mat(
            -np.sum(np.multiply(self.parents[1].value, np.log(prob + 1e-10))))
        ids = self.parents[1].value
        alpha = self.alpha[np.array(ids).reshape(-1).astype(int)]
        probs = np.mat((prob * class_mask).sum(1).view(-1, 1))
        log_p = np.log(probs)
        loss = -alpha * (np.power((1 - probs), self.gamma)) * log_p
        if self.reduction == 'mean':
            self.value = loss.mean()
        elif self.reduction == 'sum':
            self.value = loss.sum()


    def get_jacobi(self, parent):
        # 这里存在重复计算，但为了代码清晰简洁，舍弃进一步优化
        prob = SoftMax.softmax(self.parents[0].value)
        class_mask = np.mat(
            -np.sum(np.multiply(self.parents[1].value, np.log(prob + 1e-10))))
        ids = self.parents[1].value
        alpha = self.alpha[ids.data.view(-1)]
        probs = np.mat((prob * class_mask).sum(1).view(-1, 1))
        log_p = np.log(probs)
        jacobi = np.zeros_like(prob)
        if parent is self.parents[0]:
            jacobi = -alpha * self.gamma * np.power(1 - probs, self.gamma - 1) * (1 - probs) * np.log(probs)
        elif parent is self.parents[1]:
            jacobi = -alpha * self.gamma * np.power(1 - probs, self.gamma - 1) * (1 - probs) * (1 + np.log(probs))
        elif parent is self.alpha:
            jacobi = -np.power(1 - probs, self.gamma) * np.log(probs)
        return jacobi