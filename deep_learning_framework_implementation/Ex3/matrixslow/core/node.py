# -*- coding: utf-8 -*-
"""
Created on Wed Jun  5 15:23:01 2019

@author: zhangjuefei
"""
import abc #抽象基类
import numpy as np
from .graph import Graph, default_graph

class Node(object):
    """
    计算图节点类基类
    """
    
    # Node 构造函数:可用于初始化类中各种成员的实例
    # 可变参数 *parents为元组 **kargs为字典 
    def __init__(self, *parents, **kargs):

        # 计算图对象，默认为全局对象default_graph
        self.name = kargs.get('name', None)
        self.kargs = kargs
        self.graph = kargs.get('graph', default_graph)
        self.need_save = kargs.get('need_save', True)
        self.gen_node_name(**kargs)

        self.parents = list(parents)  # 父节点列表，元组不能修改，但列表可以
        self.children = []  # 子节点列表
        self.value = None   # 本节点的值
        self.jacobi = None  # 结果节点对本节点的雅可比矩阵

        # 遍历所有父节点,将本节点添加到父节点的子节点列表中,建立父子节点的双向链接
        for parent in self.parents:
            parent.children.append(self)

        # 将本节点添加到Graph类对象的节点列表中(即计算图）
        self.graph.add_node(self)

    def get_parents(self):
        """
        获取本节点的父节点
        """
        return self.parents

    def get_children(self):
        """
        获取本节点的子节点
        """
        return self.children

    def gen_node_name(self, **kargs):
        """
        生成节点名称，如果用户不指定，则根据节点类型生成类似于"MatMul:3"的节点名，
        如果指定了name_scope，则生成类似"Hidden/MatMul:3"的节点名
        """
        #实例调用__class__属性时会指向该实例对应的类
        #self.__class__.__name__实例对应类的类名
        self.name = kargs.get('name', '{}:{}'.format(
            self.__class__.__name__, self.graph.node_count())) 
        if self.graph.name_scope:
            self.name = '{}/{}'.format(self.graph.name_scope, self.name)

    def forward(self):
        """
        计算节点的值，要求所有的父节点的值都计算。
        若父节点的值未被计算，则递归调用父节点的forward方法。这是深度优先搜索方式。
        """
        for node in self.parents:
            if node.value is None:
                node.forward()

        self.compute()

    @abc.abstractmethod
    def compute(self):
        """
        抽象方法，继承Node类的具体节点需要覆盖它，根据父节点的值计算本节点的值
        """

    @abc.abstractmethod
    def get_jacobi(self, parent):
        """
        抽象方法。由于父节点是输入，计算本节点对某个父节点的雅可比矩阵，即导数/梯度。
        """

    def backward(self, result):
        """
        反向传播，计算结果节点对本节点的雅可比矩阵。
        result: 计算图中作为结果的节点，例如Loss 函数节点。
        """
        if self.jacobi is None:
            if self is result:
                self.jacobi = np.mat(np.eye(self.dimension()))
            else:
                self.jacobi = np.mat(
                    np.zeros((result.dimension(), self.dimension())))
                    
                # 遍历全部子节点，若子节点value属性为None,说明它不在本次计算路径上
                for child in self.get_children():
                    if child.value is not None:
                        # child.get_jacobi(self)：传入本节点作为父节点。
                        self.jacobi += child.backward(result) * child.get_jacobi(self)

        return self.jacobi

    def clear_jacobi(self):
        """
        清空结果节点对本节点的雅可比矩阵
        """
        self.jacobi = None

    def dimension(self):
        """
        返回本节点的值展平成向量后的维数，元素个数
        """
        return self.value.shape[0] * self.value.shape[1]

    def shape(self):
        """
        返回本节点的值作为矩阵的形状：（行数，列数）
        """
        return self.value.shape

    def reset_value(self, recursive=True):
        """
        重置本节点的值，并递归重置本节点的下游节点的值
        """

        self.value = None

        if recursive:
            for child in self.children:
                child.reset_value()


class Variable(Node):
    """
    变量节点
    """

    def __init__(self, dim, init=False, trainable=True, **kargs):
        """
        变量节点没有父节点，构造函数接受变量的形状，是否初始化以及是否参与训练的标识
        """
        Node.__init__(self,  **kargs)
        self.dim = dim

        # 如果需要初始化，则以正态分布随机初始化变量的值
        if init:
            self.value = np.mat(np.random.normal(0, 0.001, self.dim))

        # 变量节点是否参与训练
        self.trainable = trainable

    def set_value(self, value):
        """
        为变量赋值
        """
        assert isinstance(value, np.matrix) and value.shape == self.dim

        # 本节点的值被改变，重置所有下游节点的值
        self.reset_value()
        self.value = value
