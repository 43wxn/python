import torch
'''requires_grad属性为True时，autograd为改张量创建grad_fn记录其所有历史操作'''
x=torch.ones(2,2,requires_grad=True)
print(x.grad_fn)
y=x+2
print(y.grad_fn)

'''.requires_grad(True)可以改变张量的requires_grad属性'''
a = torch.randn(2, 2)
a = ((a * 3) / (a - 1))
print(a.requires_grad)
a.requires_grad_(True)
print(a.requires_grad)
b = (a * a).sum()
print(b)
print(b.grad_fn)

c=torch.tensor([[1,2],[3,4]])
print(c)
'''结果是各个位置元素的平方 不是矩阵相乘'''
print(c*c)

'''矩阵相乘是torch.mm or 用@表示乘法'''
print(torch.mm(c,c))
print(c@c)

z = y * y * 3
'''.mean方法是计算所有元素的算数平均数'''
'''为什么需要 .mean()？
    损失函数（Loss Function）：在训练神经网络时，我们通常有一个损失值来衡量模型预测的好坏。
    这个损失值需要是一个单一的标量，而不是一个向量或矩阵。优化器（如SGD, Adam）需要最小化这个标量值。
    汇总误差：例如，对于一批（batch）有4个的数据样本，模型会产生4个预测误差。z = y * y * 3 可能计算了每个样本的误差，
    而 out = z.mean() 则将这4个误差汇总成了一个代表整体性能的单一指标。
    反向传播的起点：out.backward() 只能由一个标量张量调用。
    计算平均值是常见的方法，它将所有元素的梯度“打包”成一个标量，从而可以启动反向传播过程来更新模型参数。'''
out = z.mean()
print(z, out)


x=torch.ones(2,2,requires_grad=True)
y=x+2
z=y*y*3
out=z.mean()
out.backward()  #向后传播
'''此时，z=3(x+2)^2,out=sum(z)/4,则d(out)/dx=d(z)/dx*(1/4),,,,即3(x+2)/2，当 x=1时，梯度等于4.5'''
print(x.grad)

x=torch.randn(3,requires_grad=True)
y=x*2
''' y.data：获取 y 的数值部分，而不包含梯度计算历史。这是一个常用的技巧，当你想操作张量的值但不想影响Autograd跟踪时使用。
    .norm()：计算张量的范数（norm）。对于向量来说，默认计算的是 L2范数（也就是欧几里得长度）。'''
while y.data.norm()<1000:   #y.data.norm() < 1000 非常巧妙，它的意思是：只要向量 y 的范数（长度）小于 1000，就继续将 y 的每个元素乘以 2。
    y=y*2
print(y)