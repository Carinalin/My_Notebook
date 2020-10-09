1. 在实际工作中，我们经常使用高维数据进行数据分析。但是，使用高维数据存在以下缺点：
   1. 分析困难，合理解释更困难
   2. 难以可视化
   3. 数据存储昂贵
2. PCA(principal component analysis)是一种经典的降维算法。

	# 1. Basic Statistical Properties

## 1.1. Avarage Data Point: Mean

$$
E(X) = \frac{1}{n} \sum_1^n x_n = \int_n x f_X(x)dx
$$

1. 均值描述了数据的中心(center)，但它不需要是某个特定的数据点。

2. 均值的线性转换：
   $$
   元素+同一常数C：E(X+C) = E(X)+C\\
   元素*同一常数A：E(A \cdot X) = A \cdot E(X)\\
   ---\\
   总结起来，就是：E(A \cdot X+C) = A \cdot E(X)+C
   $$
   

## 1.2. Spread of Data: Variance

1. 方差描述了一个变量的分散程度，方差公式为：
   $$
   假设\vec{X} = \{x_1,x_2,...,x_n\}\\
   则方差向量化运算为：\\
   V(X) = \frac{1}{n}\sum(X- \mu)(X-\mu)^T
   $$

2. 根据方差的公式，可以得到协方差矩阵：
   $$
   假设\vec{D} = \begin{bmatrix} x_1 & x_2 \\ y_1 & y_2 \end{bmatrix}，即每1列代表1个数据点\\
   \vec{\mu} = \begin{bmatrix} \mu_x \\ \mu_y \end{bmatrix},即每1行的均值,则协方差矩阵为：\\
   Cov(data) = \frac{1}{n}(\vec{D}- \vec{\mu})(\vec{D}-\vec{\mu})^T
   $$

3. 方差/协方差的线性转换：
   $$
   元素+同一常数C：V(X+C) = V(X)+C\\
   元素*同一常数A：V(A \cdot X) = A^2 \cdot V(X)\\
   协方差矩阵*常数A向量：Cov(A \cdot X + C) = A\ Cov(X) \ A^T
   $$
   

4. 对于相互独立的随机变量X和Y：
   $$
   V(X+Y) = V(X) + V(Y)
   $$

# 2. Vector Calculation

## 1. Dot Product

1. x的向量模/长度是：
   $$
   ||X|| = \sqrt{X_T X} = \sqrt{\sum x_i^2}
   $$

2. x和y的向量距离是：
   $$
   d(X,Y) = ||X-Y|| = \sqrt{(X-Y)^T (X-Y)}
   $$

3. x和y的点积是：
   $$
   X \cdot Y = X^T Y = \sum x_i y_i = ||X||\,\,||Y||cos{\alpha}
   $$

## 2. Inner Product

1. 内积是点积的一般化，点积是内积的一种特殊形式。内积的数学含义是对称的、正定的双线性映射(a symmetric positive definite bilinear mapping)。

   * 其中双线性(bilinear)指：两两内积成线性关系

   $$
   假设存在三个向量X,Y,Z满足(<>表示内积)：\\
   <\lambda X+Z, Y> = \lambda<X,Y> + <Z, Y>\\
   <X, \lambda Y+Z> = \lambda<X,Y> + <X,Z>
   $$

   * 正定(positive definite)是指：向量与自身的内积大于等于0。

   $$
   <X,X> ≥ 0， 且<X,X> = 0时，X=0
   $$

   * 对称性(symmetry)指：X和Y的内积等于Y和X的内积
     $$
     <X,Y> = <Y,X>
     $$
     

2. 内积的计算公式：当A为单位矩阵I时，X和Y的内积即为点积。
   $$
   <X,Y> = X^T A Y，A为线性转换矩阵(对称、正定)
   $$

3. 利用内积计算向量长度(向量模)和向量间距离：

   1. 向量模与内积公式有关，不同的内积公式会得到不同结果：

   $$
   ||X|| = \sqrt{<X,X>}\\
   相关属性：\\
   ||\lambda X| = |\lambda| \cdot ||X||\\
   ||X+Y|| ≤ ||X||+ ||Y|| 注：三角不等\\
   |<X, Y>| = ||X|| \cdot ||Y|| 注：Cauchy-Schwarz不等
   $$

   2. 向量距离和内积公式有关：
      $$
      d(X,Y) = ||X - Y|| = \sqrt{<X-Y, X-Y>}
      $$
      

4. 利用内积计算角度并理解正交(angles and orthogonality)：
   $$
   cos \alpha = \frac{<X,Y>}{||X|| \cdot ||Y||}
   $$

   * 向量间的角度告诉了我们这两个**向量的方向的相似性**。

   * 当两个向量内积为0，说明两个向量正交，角度等于90度。反过来行不通，也就是说，两个向量正交，内积不一定为0，取决于内积公式。

   * 正交基向量(orthonormal basis)：
     $$
     <b_i,b_j> = 0 ，且 i ≠ j\\
     ||b_i|| = ||b_j|| =1
     $$
     

5. 函数和随机变量的内积计算：

   * 对于函数u(x)和v(x)，其内积等于：
     $$
     <U,V> = \int_a^b u(x)v(x)dx
     $$

   * 当函数内积为0时，函数u(x)和v(x)在[a,b]内正交。

   * 对于随机变量X和Y，其协方差就是一种形式的内积，满足内积对称、正定、双线性的特点，双线性举例：
     $$
     Cov(\lambda X + Y, Z) = \lambda Cov(X,Z) + Cov(Y,Z)
     $$

   * 对于随机变量X和Y的夹角，则：
     $$
     X和Y的模为变量的标准差：||X||= \sqrt{Cov(X,X)} = \sigma(X)\\
     X和Y的夹角为：cos \theta = \frac{Cov(X,Y)}{||X|| \cdot ||Y||} = \frac{Cov(X,Y)}{\sqrt{V(X) \cdot V(Y)}}
     $$
     

## 3. K最近邻算法

1. k最近邻算法的原理基于向量间距离大小体现了向量的相似性。通过计算各个元素向量之间的距离，找到与原向量最近的k个点及其对应类别，通过投票计算选出得票最多的类别，完成预测分类。