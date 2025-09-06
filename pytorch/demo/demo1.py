import torch
# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< tensor basis operation >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# 构造一个5x3的[0，1]随机矩阵
x=torch.rand(5,3)
print(x)

# 构造一个5x3的空矩阵 不初始化
x=torch.empty(5,3)
print(x)

# 构造一个5x3的全为0的5x3矩阵 且定义元素类型
x=torch.zeros(5,3,dtype=torch.long)
print(x)

# 构造一个张量 直接使用数据
x=torch.tensor([5.5,3,1,6,7,0])
print(x)

# 创建一个tensor基于一个存在的tensor
x=x.new_ones(5,3,dtype=torch.float64) #这里new_* 方法 ones表示创建一个全是1 dtype=torch.float32的新tensor
print(x)

x=torch.randn_like(x,dtype=torch.double) # override dtype
print(x) # result has the same size of tensor

# get the dimension of tensor
y=x.size()
print(y) 

# operation
#add 
y=torch.rand(5,3)
print(y)
x=y.new_ones(5,3)
print(x)
#>>>>>>>>>>>
print(x+y)
#or
print(torch.add(x,y))
#<<<<<<<<<<<

#let a exist tensor become result
result=torch.empty(5,3)
torch.add(x,y,out=result)
print(x)
print(y)
print(result)

#add x to y
print(y)
print(x)
y.add_(x)
print(y)

# 所有使张量发生改变的方法都会有一个“_‘ 例如x.copy_(y)--change x with copy y;x.add_(y)--change x with adding y
#x.t_()--->t_() change x with tanspose x and not create a copy

#change tensor's dimension 
x=torch.rand(4,4) #tow dimension
print(x)
y=x.view(16) # one dimension
print(y)
z=x.view(-1,8) # tow dimension but change 4*4 to 2*8
print(z)

# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< tensor basis operation >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

