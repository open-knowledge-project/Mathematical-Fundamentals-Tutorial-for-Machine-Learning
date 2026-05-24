# 第5章 · Python 实现

前几章我们学习了回归、分类和评估的理论。理论理解之后，编写代码来验证和加深理解是非常有意义的。本章将使用 Python、NumPy 和 Matplotlib 手动实现核心算法——不使用 Scikit-learn 等现成的库，以便深入理解每个细节。

## 5.1 环境准备

需要安装的库：

```bash
pip install numpy matplotlib
```

或者如果你使用 Miniconda：

```bash
conda install numpy matplotlib
```

导入：

```python
import numpy as np
import matplotlib.pyplot as plt
```

## 5.2 数据标准化

在训练之前，强烈建议对输入数据进行**标准化**（Z-score Normalization）：

$$z = \frac{x - \mu}{\sigma}$$

其中 $\mu$ 是均值，$\sigma$ 是标准差。

标准化的原因：如果不同特征的数值范围差异很大（比如一个特征在 0-1，另一个在 0-10000），梯度下降的收敛会变得困难（椭圆形的等高线会导致之字形下降）。标准化使各特征的均值为 0、方差为 1，有助于更快收敛。

```python
mu = train_x.mean(axis=0)
sigma = train_x.std(axis=0)
train_z = (train_x - mu) / sigma
```

## 5.3 准备数据矩阵

为了用向量化的方式处理，需要在输入矩阵中加入全为 1 的 $x_0$ 列（对应截距项 $\theta_0$）：

```python
def to_matrix(x):
    x0 = np.ones([x.shape[0], 1])   # 全1列，对应 theta_0
    return np.hstack([x0, x])        # 水平拼接
```

对于多项式回归，还需要添加高次项：

```python
def to_matrix(x):
    x0 = np.ones([x.shape[0], 1])
    x1 = x[:, 0, np.newaxis]        # 保持二维形状
    x2 = x1 ** 2                     # 二次项
    x3 = x1 ** 3                     # 三次项
    return np.hstack([x0, x1, x2, x3])
```

对于处理线性不可分的分类问题，也需要加入高次特征（如 $x_1^2$）：

```python
def to_matrix(x):
    x0 = np.ones([x.shape[0], 1])
    x3 = x[:, 0, np.newaxis] ** 2   # 加入 x1^2
    return np.hstack([x0, x, x3])
```

## 5.4 回归的完整实现

### 批量梯度下降

```python
import numpy as np
import matplotlib.pyplot as plt

# 1. 读入数据
train = np.loadtxt('click.csv', delimiter=',', skiprows=1)
train_x = train[:, 0]      # 广告费
train_y = train[:, 1]      # 点击量

# 2. 绘制原始数据
plt.plot(train_x, train_y, 'o')
plt.show()

# 3. 标准化
mu = train_x.mean()
sigma = train_x.std()
train_z = (train_x - mu) / sigma

# 4. 构建矩阵
def to_matrix(x):
    return np.vstack([np.ones(x.shape[0]), x, x ** 2]).T

X = to_matrix(train_z)

# 5. 参数初始化
theta = np.random.randn(3)   # 随机初始化 3 个参数（theta_0, theta_1, theta_2）

# 6. 预测函数
def f(x):
    return np.dot(x, theta)

# 7. 目标函数（MSE）
def E(x, y):
    return 0.5 * np.sum((y - f(x)) ** 2)

# 8. 梯度下降
ETA = 1e-3    # 学习率
epoch = 5000  # 重复次数

errors = []   # 记录误差变化
for _ in range(epoch):
    theta = theta - ETA * np.dot(f(X) - train_y, X)
    errors.append(E(X, train_y))

# 9. 绘制学习曲线
plt.plot(errors)
plt.xlabel('Epoch')
plt.ylabel('MSE')
plt.show()

# 10. 绘制拟合结果
x_line = np.linspace(-2, 2, 100)
X_line = to_matrix(x_line)
y_line = f(X_line)

plt.plot(train_z, train_y, 'o')
plt.plot(x_line, y_line)
plt.show()

print('训练完成, theta =', theta)
```

**关键解释**：`np.dot(f(X) - train_y, X)` 这一行同时计算了所有参数的梯度（利用 NumPy 的向量化运算）。对于 $\theta_0$、$\theta_1$、$\theta_2$ 三个参数，这个表达式产生的向量分别为：$\sum[\text{error}]$、$\sum[\text{error} \cdot x_1]$、$\sum[\text{error} \cdot x_2]$。

### 随机梯度下降法（SGD）

只需将学习部分替换为：

```python
# SGD 学习
for _ in range(epoch):
    p = np.random.permutation(X.shape[0])   # 随机排列索引
    for x, y in zip(X[p, :], train_y[p]):   # 逐个样本
        theta = theta - ETA * (f(x) - y) * x
```

注意这里 `(f(x) - y) * x` 是针对**单个样本**的梯度，而不是所有样本的梯度之和。

**随机排列的重要性**：如果不能随机排列，而是按固定顺序遍历数据，可能导致参数更新存在系统性偏差。

## 5.5 分类的完整实现（逻辑回归）

### 线性可分的情况

```python
import numpy as np
import matplotlib.pyplot as plt

# 1. 读入数据
train = np.loadtxt('images2.csv', delimiter=',', skiprows=1)
train_x = train[:, 0:2]
train_y = train[:, 2]

# 2. 标准化
mu = train_x.mean(axis=0)
sigma = train_x.std(axis=0)
train_z = (train_x - mu) / sigma

# 3. 构建矩阵（加入 x0 = 1）
def to_matrix(x):
    x0 = np.ones([x.shape[0], 1])
    return np.hstack([x0, x])

X = to_matrix(train_z)

# 4. 初始化参数
theta = np.random.rand(3)

# 5. sigmoid 函数
def f(x):
    return 1 / (1 + np.exp(-np.dot(x, theta)))

# 6. 分类函数
def classify(x):
    return (f(x) >= 0.5).astype(np.int64)

# 7. 梯度下降（注意：公式与回归相同！）
ETA = 1e-3
epoch = 5000
for _ in range(epoch):
    theta = theta - ETA * np.dot(f(X) - train_y, X)

# 8. 绘制决策边界
x0 = np.linspace(-2, 2, 100)
plt.plot(train_z[train_y == 1, 0], train_z[train_y == 1, 1], 'o', label='Positive')
plt.plot(train_z[train_y == 0, 0], train_z[train_y == 0, 1], 'x', label='Negative')
plt.plot(x0, -(theta[0] + theta[1] * x0) / theta[2], linestyle='dashed', label='Decision Boundary')
plt.legend()
plt.show()

print('训练完成, Accuracy =', (classify(X) == train_y).mean())
```

**决策边界的绘制**：$\theta^T x = 0$ 展开是 $\theta_0 + \theta_1 \cdot x_1 + \theta_2 \cdot x_2 = 0$，解得 $x_2 = -(\theta_0 + \theta_1 \cdot x_1) / \theta_2$。

### 线性不可分的情况

关键在于在 `to_matrix` 中加入高次项：

```python
def to_matrix(x):
    x0 = np.ones([x.shape[0], 1])
    x3 = x[:, 0, np.newaxis] ** 2
    return np.hstack([x0, x, x3])

# 参数数量变为 4（增加了一个 theta_3 对应 x1^2）
theta = np.random.rand(4)
X = to_matrix(train_z)

# sigmoid 函数和训练过程完全不变
def f(x):
    return 1 / (1 + np.exp(-np.dot(x, theta)))

ETA = 1e-3
epoch = 5000
for _ in range(epoch):
    theta = theta - ETA * np.dot(f(X) - train_y, X)

# 决策边界变为二次曲线
x1 = np.linspace(-2, 2, 100)
x2 = -(theta[0] + theta[1] * x1 + theta[3] * x1 ** 2) / theta[2]

plt.plot(train_z[train_y == 1, 0], train_z[train_y == 1, 1], 'o')
plt.plot(train_z[train_y == 0, 0], train_z[train_y == 0, 1], 'x')
plt.plot(x1, x2, linestyle='dashed')
plt.show()
```

### 绘制学习曲线（Accuracy vs Epoch）

```python
theta = np.random.rand(4)
accuracies = []

for _ in range(epoch):
    theta = theta - ETA * np.dot(f(X) - train_y, X)
    result = classify(X) == train_y
    accuracy = result.mean()
    accuracies.append(accuracy)

plt.plot(accuracies)
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.show()
```

## 5.6 正则化的实现

正则化只需修改参数更新步骤，加入正则化项：

```python
# 无正则化
theta = theta - ETA * np.dot(f(X) - train_y, X)

# 加入 L2 正则化
LAMBDA = 1e-2  # 正则化参数
theta = theta - ETA * (np.dot(f(X) - train_y, X) + LAMBDA * theta)
```

等价于：

```python
theta = (1 - ETA * LAMBDA) * theta - ETA * np.dot(f(X) - train_y, X)
```

这体现了"权重衰减"——每次更新前先把参数缩小一点。

## 5.7 SGD 的分类实现

分类问题同样可以使用 SGD：

```python
theta = np.random.rand(4)

for _ in range(epoch):
    p = np.random.permutation(X.shape[0])
    for x, y in zip(X[p, :], train_y[p]):
        theta = theta - ETA * (f(x) - y) * x
```

注意这里的数据是标准的 NumPy 数组，`f(x)` 返回的是标量（因为 `x` 是单个样本），而 `(f(x) - y) * x` 产生的是与 `x` 同维度的梯度向量。

## 5.8 完整的回归与 SGD 代码

```python
import numpy as np
import matplotlib.pyplot as plt

# ===== 数据加载与标准化 =====
train = np.loadtxt('click.csv', delimiter=',', skiprows=1)
train_x = train[:, 0]
train_y = train[:, 1]

mu = train_x.mean()
sigma = train_x.std()
train_z = (train_x - mu) / sigma

# ===== 构建设计矩阵 =====
def to_matrix(x):
    return np.vstack([np.ones(x.shape[0]), x, x ** 2]).T

X = to_matrix(train_z)

# ===== 模型训练 =====
def train_model(X, y, method='sgd', eta=1e-3, epoch=5000):
    theta = np.random.randn(X.shape[1])
    
    for _ in range(epoch):
        if method == 'batch':
            theta -= eta * np.dot(np.dot(X, theta) - y, X)
        else:  # sgd
            p = np.random.permutation(X.shape[0])
            for xi, yi in zip(X[p, :], y[p]):
                theta -= eta * (np.dot(xi, theta) - yi) * xi
    
    return theta

theta = train_model(X, train_y, method='sgd')
print('theta:', theta)

# ===== 预测与绘图 =====
def predict(x):
    X_new = to_matrix(np.array([(x - mu) / sigma]))
    return np.dot(X_new, theta)

# 绘制结果
x_plot = np.linspace(train_x.min(), train_x.max(), 100)
y_plot = [predict(x) for x in x_plot]

plt.plot(train_x, train_y, 'o', label='Training Data')
plt.plot(x_plot, y_plot, '-', label='Fitted Curve')
plt.xlabel('Ad Spend (Standardized)')
plt.ylabel('Clicks')
plt.legend()
plt.show()


