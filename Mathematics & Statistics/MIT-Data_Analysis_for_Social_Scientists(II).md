[toc]

# Module 5

> Special Distributions, the Sample Mean, the Central Limit Theorem(特殊分布、样本均值、中心极限定理）
> 常用名词：
>
> * IID:Independent Identically Distributed/独立同分布，即变量之间相互独立且满足同一分布如二项分布
> * CDF
> * PDF
> * underlying distribution: 底层分布，即总体分布，但在统计学上，我们永远不可能采集到总体（总体是无尽的），因此，我们需要去估计总体的参数。

## 1. Special Distributions

> * 特殊分布的重要性在于：1. 他们之间以有用的方式相互联系。2. 他们为各种随机现象构建模型。
> * 事实上总会有新的候选特殊分布出现，不过我们了解经典的特殊分布即可，他们都有标准的公式和指定的参数。
> * 经典的特殊分布：
>
> 1. 离散概率分布：**伯努利、二项、负二项、几何、泊松**等。
> 2. 连续概率分布：**连续型均匀分布、指数、正态、对数正态、帕累托**等。

### 1.1. Bernoulli Distribution（伯努利分布）

* 伯努利试验是只有两种可能结果（成功或失败）的**单次随机试验**，而其结果就是伯努利分布。

* X的取值仅为0或1(success or failure)，对应的概率为p和q(q = 1-p)。

* X的期望值E(X)为p，方差V(X)为pq。

* 概率f(x; p)为
  $$
  f(x\,; p) = p^{x}q^{1-x} \quad for \quad x ∈ \{0, 1\}
  $$

* 经典案例（单次试验）：
  1. 一个硬币掉落后是人头朝上吗？
  2. 掷一个骰子正面会是6吗？
  3. 一个可能是顾客的人会买我的产品吗？
     ![img](MIT-Data_Analysis_for_Social_Scientists(II).assets/65DC3942-6904-4AB5-949C-A4AC39E39154.jpg)

### 1.2. Binomial Distribution（二项分布）

* **二项分布是多次进行伯努利试验的结果**，实际上，当n = 1时，二项分布就是伯努利分布。

* X表示在n个独立实验中成功的次数，要求：事件之间相互独立且出现概率相同(均为p)。

* X的期望值E(X)为np，方差V(X)为npq

* n次试验中正好得到k次成功的概率为
  $$
  f(k\,; n, p) =  P(X=k) = \frac{n!}{x!(n-x)!}p^{x}(1-p)^{n-x}
  $$

* 经典案例（二项试验）：
  1.射击10次，射中2次的概率是多少？
  ![img](MIT-Data_Analysis_for_Social_Scientists(II).assets/7D3EEABE-9C77-43CF-9F11-44FA96CAF803.jpg)

```
# 1. 创建n个X~B(size,prob)二项分布的随机变量
bino <- rbinom(1000, 8, 0.2)#1000表示观察数n，8表示试验数size，0.2表示成功概率prob

# 2. 计算X~B(size,prob)二项分布某个值x的概率
dbinom(4, 10, 0.1)#4表示x，10表示试验数size，0.1表示成功概率prob

# 3. 计算X~B(size,prob)二项分布某个值x的累积概率
pbinom(4, 10, 0.1, lower.tail = TRUE)#4表示x，10表示试验数size，0.1表示成功概率prob，lower.tail = TRUE表示计算P(X≤x)。

# 4. 计算二项分布特定累积p概率对应的值x
qbinom(0.25, 10, 0.2, lower.tail = TRUE)#0.25表示累积概率p，10表示试验数size，0.2表示成功概率prob。
```

### 1.3. Negative Binomial Distribution（负二项分布）

* 负二项分布描述在一系列独立同分布的伯努利试验中，成功次数到达指定次数（记为r）时失败次数的离散概率分布，又称为巴斯卡分佈（Pascal distribution）。比如，如果我们定义掷骰子随机变量x值为x=1时为成功，所有x≠1为失败，这时我们反复掷骰子直到1出现3次（成功次数r=3），此时非1数字出现次数的概率分布即为负二项分布。

![img](MIT-Data_Analysis_for_Social_Scientists(II).assets/0C870CDF-2FD3-4AFD-BFD5-0AE75C73DF02.jpg)
![img](MIT-Data_Analysis_for_Social_Scientists(II).assets/5D5413FC-9E3D-4370-9C7C-9B8783BCF76F.jpg)

### 1.4. Hypergeometric distribution（超几何分布）

* 和二项分布不一样的是，超几何分布描述了由N个物件中抽出n个物件，成功抽出指定种类的物件的个数（without replacement）。当n等于1时，超几何分布就是伯努利分布。

* A表示成功次数，B表示失败次数，N表示总次数（N=A+B）,n表示抽取个数，X表示抽取中成功的次数，则概率为：
  $$
  p=\frac{A}{A+B} \qquad q=\frac{B}{A+B}\\
  f(X|A, B, n) = \frac{\begin{pmatrix}A\\x\end{pmatrix}\begin{pmatrix}B\\n-x\end{pmatrix}}{\begin{pmatrix}A+B\\n\end{pmatrix}}\\
  \text{(注：从A中取x次乘以从B中取n-x次除以从N中取n次)}
  $$

* 注意：0! = 1。

```
#R的计算
dhyper(x,A,B,n)
```

* X的期望值$E(X)=\frac{nA}{A+B}$，方差$V(X) = n{\frac{A}{A+B}}{\frac{B}{A+B}}{\frac{A+B-n}{A+B-1}}$。
* 与二项分布的关系：
  1. 当N比n大得多的时候，我们可以忽略不放回的n，这时超几何分布近似为二项分布。
* 经典案例（不放回抽样）：
  1. 从一个有红球和黑球的箱子里一次性取出n个球，其中k个红球的概率？
     ![img](MIT-Data_Analysis_for_Social_Scientists(II).assets/3DFB4652-A140-41AD-8B6C-9A942D781814.jpg)
     ![img](MIT-Data_Analysis_for_Social_Scientists(II).assets/81D52D1E-8EDE-4E6D-B6A7-7EB1555C9DF2.jpg)
     ![img](MIT-Data_Analysis_for_Social_Scientists(II).assets/4AA30173-A3AE-44DB-8E61-D184C14D4BF3.jpg)

### 1.5. Poisson Distribution（泊松分布）

* 泊松分布适合于描述

  单位时间内随机事件发生的次数的概率分布

  ，经典案例有：

  1. 某一服务设施在一定时间内受到的服务请求的次数
  2. 电话交换机一定时间内接到呼叫的次数
  3. 一场足球比赛的进球次数等。

* 泊松分布描述了给定数量的事件在一定时间内发生的概率，泊松分布建模的三个条件有：

  1. 事件之间相互独立
  2. 事件是可计量的(countable)
  3. 已知该段时间内的平均发生概率。

* 对于非负整数k，泊松分布的概率为：
  $$
  P(N_t = k) = \frac{{(\gamma t)}^k e^{-\gamma t}}{k!} \qquad \lambda = \gamma t\\
  \text{注：$\gamma \,$表示单位时间内的到达率，$t \,$表示时间单位数，$\lambda \,$表示一定时间内的到达率。}
  $$

* 期望值和方差均为$\lambda \,$。
  ![img](MIT-Data_Analysis_for_Social_Scientists(II).assets/0F65DD40-3666-46ED-A3DB-C94EBE2B6948.jpg)

* 泊松分布图特征：

  1. 泊松分布图是倾斜的(因为X不可能为负)，不过随着$\lambda \,$的增大会越来越对称。

     ```
     # 计算泊松分布的概率
     ppois(1, lambda = 7, lower = FALSE) # P(N > 1)
     ```

### 1.6. Exponential Distribution（指数分布）

* 指数分布可以用来表示独立随机事件发生的时间间隔（泊松分布中两个事件的时间间隔），经典案例有：

  1. 旅客进入机场的时间间隔
  2. 打进客服中心电话的时间间隔
  3. 维基百科新条目出现的时间间隔

* 指数分布的概率为：
  $$
  f_x = \lambda e^{-\lambda x} \qquad x>0
  $$

* 期望值和方差分别为：
  $$
  E(X) = \frac{1}{\lambda}\qquad V(X) = \frac{1}{\lambda^2}
  $$

* 指数分布具有**无记忆性**的特点(memoryless)，即当事件还未发生时，事件发生的概率不会受时间影响（即P(t=0)=P(t=1)=...）。举个例子：足球比赛中，x表示进球的时间间隔，如果还未进球，那么在15分钟时的进球概率和50分钟时的是一样的。
* 在R中，用rexp()可以生成指数分布的随机数：

```
# plot exponential distribution
pdf("random from exponential.pdf")
y_rexp <- rexp(10000, rate =3)
density_y_rexp <- density(y_rexp)
plot(density_y_rexp，main = "exponential distribution", lwd = 3, col = "darkred", xlab="")
hide <- dev.off()
```

![img](MIT-Data_Analysis_for_Social_Scientists(II).assets/3C62E843-4A48-40DA-BADB-747AF60BAD03.jpg)

### 1.7. Uniform Distributions（均匀分布）

* 均匀分布描述的是在特定区间内概率相同的情况，称X服从[a, b]上的均匀分布，记为X~U(a, b)。

* 概率密度函数为：
  $$
  fx = \frac{1}{b-a} \qquad for \quad a ≤ x ≤ b
  $$

* 期望值和方差分别为：
  $$
  E(X) = \frac{a+b}{2}\qquad V(X) = \frac{({b-a})^2}{12}
  $$

![img](MIT-Data_Analysis_for_Social_Scientists(II).assets/E23D946B-00B6-4EF5-9FF6-26DEC33C8CDF.jpg)

* 将a设为0，b设为1，则称为得到标准均匀分布(standard uniform distribution)。
* 均匀分布的重要应用：**假设检验中伪随机数的生成**。很多编程语言都是利用标准均匀分布生成的伪随机数。
  ![img](MIT-Data_Analysis_for_Social_Scientists(II).assets/96D843BB-2830-40CB-957F-3A0B3D1E44DC.jpg)
* 均匀分布的相关R函数

```
# 计算概率密度
dunif(x, min = 0, max = 1, log = FALSE)
# 计算分布函数
punif(q, min = 0, max = 1, lower.tail = TRUE, log.p = FALSE)
# 得到分位数函数
qunif(p, min = 0, max = 1, lower.tail = TRUE, log.p = FALSE)
# 随机生成均匀分布的数字
runif(n, min = 0, max = 1)
```

### 1.8. Normal Distributions（正态分布）

* 正态分布和二项分布的关系：

  1. **二项分布是离散分布，而正态分布是连续分布**.
  2. **当二项分布的n值（试验次数）趋向于无穷大时，二项分布近似可以看成正态分布**。
  3. **正态分布的图像是一个钟形曲线，而二项分布的图像为直方图，直方图的顶端可以近似连接成为一条钟形曲线**。

* 假设随机变量$X = \mu + \sigma Z \quad \sigma≠0$，其中Z是标准正态变量($\mu_Z = 0 , \sigma_Z =1$)，则正态分布$X \sim N(\mu , \sigma^2)$的概率密度函数是：
  $$
  f(x| \mu, \sigma) = \frac{1}{\sigma}\phi \left(\frac{x-\mu }{\sigma} \right) = \frac{1}{\sqrt{2 \pi } \sigma} e^{-\frac{1}{2}\left(\frac{x-\mu }{\sigma} \right)^2}
  $$

* 期望值和方差分别为：
  $$
  E(X) = E(Z)+\mu =\mu \qquad
  V(X) = \sigma^2 * V(Z) = \sigma^2
  $$

* 正态分布/IID的属性：
  $$
  \text{如果} X_1 \text{满足正态分布，则} X_2 = a+bX_1 \text{同样满足正态分布，且：} \\
  E(X_2) = a+bE(X_1) \qquad V(X_2) = b^2V(X_1)
  $$

* 正态分布/IID标准化(standardization)：
  $$
  Z = \frac{x-\mu}{\sigma}
  $$

![img](MIT-Data_Analysis_for_Social_Scientists(II).assets/EFBBB5DE-B608-4EA9-B280-9FDDB052053E.jpg)

* 正态分布曲线的面积/积分计算：

```
# 求特定分位点的面积
p <- pnorm(q, mean = 0, sd = 1, lower.tail = TRUE, log.p = FALSE)

# 求特定面积的分位点(pnorm的逆运算)
q <- qnorm(p, mean = 0, sd = 1, lower.tail = TRUE, log.p = FALSE)

# 生成符合正态分布的指定数量的变量
norm <- rnorm(n, mean = 0, sd = 1)

# 得到指定点的概率密度
dnorm(x, mean = 0, sd = 1, log = FALSE)
```

### 1.9. Geometric distribution(几何分布)

1. 几何分布指在伯努利试验中，得到第一次成功所需要的试验次数X，记为：X~G(X)，则在得到第一次成功之前所经历的失败次数是X−1 次，几何分布的公式为：

   $P(X=k) = (1-p)^{k-1} p$

2. 几何分布的期望值和方差分别为：

   $E(X) = \frac{1}{p}$

   $E(X) = \frac{1-p}{p^2}$![img](MIT-Data_Analysis_for_Social_Scientists(II).assets/B6DEB3FA-238B-46B0-9B1B-CD5F5BC45369.jpg)

## 2. the Sample Mean

### 2.1. conception

> The sample mean is the arithmetic average of the random variables, but also describes the arithmetic average of the realization of those random variables.

* 样本均值是n个**随机变量/实现**(random variables/realizations)的算术平均值，用符号$\overline{x}_n$表示，用于描述样本分布。
* 样本均值的属性：**变量$X_i$为来自随机样本的随机变量，因为样本均值其实是随机变量的一个函数，则样本均值$\overline{x}_n$也是一个随机变量**。
  ![img](MIT-Data_Analysis_for_Social_Scientists(II).assets/7F5515B7-A9EE-4A1C-8D27-6FFAAC489605.jpg)

### 2.2. expectation and variance

* 假设所有的随机变量满足同一个独立恒等的分布，该分布的期望值为$\mu$，方差为$\sigma^2$，则样本均值的期望值和方差为：
  $$
  E(\overline{x}_n) = E(\frac{1}{n}\sum x_i) = \frac{1}{n} \sum E(X_i) = \frac{1}{n} \sum \mu = \frac{1}{n} * n * \mu = \mu\\
  V(\overline{x}_n) = V(\frac{1}{n}\sum x_i) = \frac{1}{n^2} \sum V(X_i) = \frac{1}{n^2} \sum \sigma^2 = \frac{\sigma^2}{n}
  $$

* 样本均值的期望值和方差特点：
  * 方差计算要求独立性，期望值计算不要求。
  * 样本均值的分布：期望值和随机变量相同，但方差更小。随着n的增大，分布更集中。

## 3. The Central Limit Theorem

* 中心极限定理(CLT)被认为是统计学的基础，其规定：**当n趋向于无穷大时，标准化的随机变量样本均值概率分布趋向于标准正态分布**。其实也就是说，当n趋向无穷时，任何IID的样本均值近似于正态分布($\text{mean} = \mu, \text{std} =\frac{\sigma}{\sqrt n})$。
* 实际上，需要注意的是，变量不是随机的也符合中心极限定理。
* 根据中心极限定理，在样本容量很大时，总体的抽样分布是趋向于正态分布的，最终都可以依据正态分布的检验公式对它进行下一步分析。该定理在理论上保证了我们可以用只抽样一部分的方法，达到推测研究对象统计参数的目的。
* 中心极限定理的作用：**用样本数据估计总体参数**。
* 这个定理也说明了为什么正态分布那么重要，为什么我们经常说数据分布呈正态有利于进一步分析。
  ![img](MIT-Data_Analysis_for_Social_Scientists(II).assets/DDDA0BEB-9ADD-4F74-BAD1-BED0A5F5E279.jpg)
  ![img](MIT-Data_Analysis_for_Social_Scientists(II).assets/5530B017-8F3A-4D73-A652-7D15ACCB0627.jpg)

## 4. study of estimation and inference

> 统计学(Statistics)是关于估计和推断的研究(estimation and inference)。

### 4.1. some definition

1. 估计相关定义：
   1. 估计(estimation)是试图确定一个分布的具体参数，因为这将给我们很多关于分布形状的信息。例如，对于正态分布，估计通常试图确定平均值和方差。
   2. 估计量(estimator)指随机样本中随机变量的特定函数(function of the random sample)，用于展示随机变量分布的有用属性(useful properties)，如样本均值(sample mean)，样本数量(sample size)等。
   3. 估计值(estimate)指运用估计量计算得到的结果，即特定函数的实现(realization)。我们一般不区分估计量和估计，都用$\widehat{\theta}$表示。
2. 参数(parameter)：
   1. 指索引分布族的常数(constant indexing a family of distributions)，一般用$\theta$表示，如正态分布的$\mu$和$\sigma^2$；指数分布的$\lambda$；均匀分布的a、b；二项分布的n、p等。
   2. 也就是说，你知道数据是哪种分布(the family of distributions)，且该分布的相关参数值，则可识别唯一的分布。
   3. 很多时候，我们想要找到观测的随机过程或现象的参数，即估计未知参数，这时我们一般用$\widehat{\theta}$来表示这个估计值。
3. 随机变量(random variables)：关于随机变量，我们需要同时记住其两个概念(整体与个体)：
   1. 数学构造(mathematical construct)：一个从样本空间到实数的数学构造，一般用*X*表示随机变量。
   2. 实现(realization)：以不同的可能性实现不同的随机结果，一般用*x*表示可能的结果(realization)
4. 随机样本(random sample)：指独立同分布的随机变量的集合，简单一点说，就是数据(data)。

### 4.2. accessing and choosing estimators

* 一个估计量也是一个随机变量，也就是说，估计量也有着自己的分布。因此，评估估计量要基于分布的特征。
* 根据中心极限定理，当n足够大时，估计量的分布近似于正态分布。
* 无偏估计unbiased)：如果结果$\Theta$中所有的$\theta$满足$E(\widehat{\theta}) = \theta$，则认为关于$\theta$的估计是无偏的。
  ![img](MIT-Data_Analysis_for_Social_Scientists(II).assets/7E381982-AEC0-4C4B-A6E4-BE357E2E9923.jpg)
  ![img](MIT-Data_Analysis_for_Social_Scientists(II).assets/A275C49E-D33F-403A-B16E-8FFA6FC06A33.jpg)

# Module 6

## 1. Assessing and Deriving Estimators

### 1.1. Assessing Estimators

> 选择估计量的标准有：无偏性(
> unbiasedness)、有效性(efficiency)、一致性(consistancy)、鲁棒性(robustness)。

1. 无偏性(unbiasedness)：如果结果$\Theta$中所有的$\theta$满足$E(\widehat{\theta}) = \theta$，则认为关于$\theta$的估计是无偏的。

   * 定理1：用iid样本的样本均值估计总体均值是无偏的。
   * 定理2：用iid样本的样本方差估计总体方差是无偏的，样本方差为：$S^2 = \frac{1}{n-1} \sum (x_i - \overline {x_n})^2$

2. 有效性(efficiency)：给定两个无偏估计量，$\widehat{\theta_1}$和$\widehat{\theta_2}$，当$V(\widehat{\theta_1}) < V(\widehat{\theta_2})$时，我们认为$\widehat{\theta_1}$比$\widehat{\theta_2}$更有效。

   * 怎么平衡有效性和无偏性呢? 方法之一：均方误差(MSE, mean squared error) ：
     $$
     MSE(\widehat {\theta}) = E[(\widehat {\theta} - {\theta})^2] = V(\widehat {\theta}) + [E(\widehat {\theta}) - {\theta}]^2
     $$

![img](MIT-Data_Analysis_for_Social_Scientists(II).assets/8FB1FE14-6CA2-4257-9777-00E30CCA4390.jpg)

1. 一致性(consistent)：简单来说，如果一个估计量的分布随着n趋向于无穷，折叠为参数实际值附近的一个点时，则称估计量是一致的。
   $$
   \lim_{n\rightarrow+\infty} P(|\theta - \widehat \theta_n| <\delta) = 1 \qquad  \delta \text{为某个特定的值}
   $$

![img](MIT-Data_Analysis_for_Social_Scientists(II).assets/DFC20E52-1538-4616-B383-90F9F1A8A5CC.jpg)

1. 鲁棒性(robustness)：除了以上三个标准外，选择估计量还应考虑：估计量的计算有多简单？我们的假设有多可靠(eg: 如果我们假设了错误的分布，估计量能否正常计算）。
   * 一个估计量如果能很好地估计参数即使我们对总体分布的假设上犯了错误，这说明它是稳健的。
   * 估计量可能对一种错误假设可靠，但对另一种错误假设不可靠。
     ![img](MIT-Data_Analysis_for_Social_Scientists(II).assets/975942E4-7A58-4F6C-8EB2-93472E776E2B.jpg)

### 1.2. Deriving Estimators

> 推导估计量的框架有：矩量法(the Method of Moments)、最大似然法(Maximum Likelihood Estimation)

1. 矩估计法(MM)：1894年由数理统计之父karl pearson提出。矩估计基于样本均值的无偏估计和k阶矩原理提出：

   1. 假设我们有k个待估参数，用参数表示总体(population moments)的1阶矩(总体均值）、2阶矩(平方总体均值）、直到k阶矩(k方总体均值)，我们就得到了k个方程，k个未知量（待估参数）；
   2. 解得每个待估参数，接着用样本k阶矩替换总体k阶矩即完成估计。
      ![img](MIT-Data_Analysis_for_Social_Scientists(II).assets/C84E9D8C-3410-4B15-A15E-7206F2625449.jpg)

2. 最大似然估计(MLE)：最大似然估计可以说是应用非常广泛的一种参数估计的方法。它的原理也很简单：**利用已知的样本，找出最有可能生成该样本的参数**。
   ![img](MIT-Data_Analysis_for_Social_Scientists(II).assets/CA507B3D-BA8F-4D9D-AEAC-962EA9628223.jpg)

   1. 概率和似然性的区别：
      * 概率$P(x| \theta)$是在已知参数$\theta$的情况下，发生观测结果为x的可能性；
      * 似然性$L( \theta | x)$则是从观测结果x出发，参数$\theta$的可能性分布。最大似然就是取可能性最大的$\theta$。

   2. 最大似然估计：对于给定的观测数据x，我们希望能从所有的参数$\theta_1, \theta_2, ... , \theta_n$中找出能最大概率生成观测数据的参数$\theta^*$作为估计结果。
      $$
      \theta^* = argmax P(x| \theta)
      $$

   ![img](MIT-Data_Analysis_for_Social_Scientists(II).assets/26CCDEF2-9D43-4F82-A180-F14060AC9598.jpg)

   3. 最大似然估计量：首先得到似然函数（因为变量之间是相互独立的，因此得到是每个变量概率密度的乘积/the joint PDF of our random sample），然后对数求导为0(对数可以减少计算量)，得到用样本属性(eg: $\mu, \sigma^2$)表示的$\theta$的最大值。
      $$
      L( \theta | x) = \prod^{n} f(x|\theta)
      $$

      * 些似然函数在最大值时是不可微的，此时不能用导数求最大值，而是其他取巧方式。如均为分布$X \sim U(0,\theta )$，用最大似然法估计$\theta$就没办法求导。
        ![img](MIT-Data_Analysis_for_Social_Scientists(II).assets/F18BB578-B7E8-4115-89E7-58D4669F9F0C.jpg)

3. 一些用最大似然估计法的例子：

   1. 均匀分布$X \sim U(0, \theta)$
      ![img](MIT-Data_Analysis_for_Social_Scientists(II).assets/65A179A4-E780-4B07-B721-ED063F560AC9.jpg)
   2. 均匀分布$X \sim U(\theta - \frac{1}{2}, \theta + \frac{1}{2})$
      ![img](MIT-Data_Analysis_for_Social_Scientists(II).assets/99B7FD9D-E73B-4E0D-A043-71B253CFE590.jpg)
      ![img](MIT-Data_Analysis_for_Social_Scientists(II).assets/5BA46402-BD1D-4542-9594-0AA37EBFEF23.jpg)

4. 最大似然法的属性（propertites）：

   1. 如果在具有一致性的估计量中有一个有效估计量，MLE会选择它。
   2. 在一定规则的条件下，MLEs分布呈现渐进正态分布(中心极限定理(CLT)在MLE的应用)。
   3. MLE有以下缺点：MLEs分布可能是偏斜的(如max{x})；MLE可能计算困难；MLE鲁棒性低，如果对总体分布(underlying/population distribution)估计错误，可能会造成较大的偏差。
      ![img](MIT-Data_Analysis_for_Social_Scientists(II).assets/FDC3232C-FDEA-495A-B545-3D83804B0CA6.jpg)

## 2. Estimator Reliability, Hypothesis Testing etc.

### 2.1. Estimator Reliability

> 在进行估计量展示时，我们需要通过一些指标告诉利益相关者估计量的可信度，常用的指标有**估计量分布的方差(variance)、标准误差(standard error)和置信区间(confidence interval)**。

1. 方差：我们可以用估计方差(estimated variance)来量化可信度，方差越大，说明该估计量的分布越分散，误差可能越大。

2. 标准误差(standard error/standard deviation)：一个估计量的标准差就是方差的平方根，评估方法和方差一致。比如，样本均值的分布方差为$\frac{\sigma^2}{n}$，则我们可以知道：
   $$
   SE(\overline{X_n}) = \frac{\sigma}{\sqrt {n}} = \frac{\hat \sigma}{\sqrt {n}}
   $$

* 根据该公式可知：随着样本量n的增大，样本均值的期望标准误差会减小。

3. 置信区间(confidence interval，CI)：和之前的点估计(point estimate)不同，区间估计在一定置信度($1- \alpha$)下，告诉我们估计量所在的分布区间，即满足$P(\hat{\theta_1}≤ \theta ≤ \hat{\theta_2}) = 1- \alpha$。不过**置信区间所表达的信息和标准误是相同的**，因为置信区间是通过标准误计算得来的。
   1. 计算置信区间：$[\overline{x} + Z_{\frac{\alpha}{2}}* (\hat{\sigma}/ \sqrt{n}), \overline{x} - Z_{\frac{\alpha}{2}}* (\hat{\sigma}/ \sqrt{n})]$,Z值指正态分布(n＜30，t分布)对应$2/\alpha$处的值(eg：95%的置信区间即0.025对应的Z值为-1.96)，$\hat{\sigma}$指样本标准差(也写成S)，$\hat{\sigma}/ \sqrt{n}$指样本均值的估计标准差。

### 2.2. Chi Square, t and F distribution

1. 卡方分布(chi square, $\chi^2$)、学生t分布(student)、F分布不像二项分布等可用于描述现实中发生的随机现象，它们**用于描述估计量的分布，以便构建置信区间和进行假设检验**(distribution of estimators or functions of estimators)。
2. 卡方分布：用于描述样本方差的分布。所有满足iid样本的方差都符合卡方分布(无论样本是什么分布)。
   1. 无偏的样本方差公式：$S^2 = \frac{1}{n-1} \sum (x_i - \overline{x})^2$
   2. 卡方分布：$\frac{(n-1)S^2}{\sigma^2} \sim \chi^2_{n-1}$，卡方分布只有一个参数n-1，n-1又叫做自由度(degrees of freedom)。
3. 用于描述样本均值的分布。根据中心极限定理，我们知道当n趋向于$\infty$，样本均值满足$\frac{\sqrt{n}(\overline{x} - \mu)}{\sigma} \sim N(0,1)$。由于总体方差是不可知的，t分布将总体方差$\sigma^2$替换成了样本方差$S^2$。
   1. t分布：$\frac{\sqrt{n}(\overline{x} - \mu)}{S} \sim t_{n-1}$
   2. 当n越来越大时，样本方差趋向于总体方差，t分布趋向于标准正态分布(The student’s t-distribution converges to the normal distribution as 𝑛 increases)。
   3. t分布和正态分布相比，尾部更肥，方差更大。因此，当n较小且不知道总体方差时，意味着我的信心越少，t分布的置信区间会越大。![img](MIT-Data_Analysis_for_Social_Scientists(II).assets/D6FC46D1-1901-4FDD-A278-D9F48A9EA3F9.jpg)
4. F分布：用于描述两个分布是否同方差(ps：总体方差)，检验两个变量的分布是否相同。
   1. 举例：假设变量X1的样本方差卡方分布为$X_1$，变量X2的样本方差卡方分布为$X_2$，则两者的比值$\frac{X_1}{X_2}$符合自由度为$n_1 -1$，$n_2 -1$的F分布。
   2. F分布：$X_1 \sim \chi^2_{n_1-1}$，$X_2 \sim \chi^2_{n_2-1}$，则$\frac{X_1/{n_1-1}}{X_2/{n_2-1}} \sim F_{n_1-1,n_2-1}$

### 2.3. Hypothesis Testing

1. 统计学家利用假设检验来回答各种各样的问题，并量化我们对问题的信心(quanlify how confident we are in the answers)，其目的是解释对从总体中抽取的样本是否有足够证据能断言总体的一些特征。

2. 假设检验的相关定义：

   1. 假设(hypothesis)：关于一个随机变量总体分布的设想。

   2. 维持的假设(maintained hypothesis)：指不能或不会被检验的假设。

   3. 可检验假设(testable hypothesis)：能够通过样本被检验的假设。

   4. 零假设(null hypothesis)：指将进行检验的假设，记为$H_0$。

   5. 备择假设alternative hypothesis)：指除了零假设之外的所有可能性，记为$H_A$。

3. 一类错误$\alpha$和二类错误$\beta$：

   1. 显著性水平(significance level)：即零假设为真却被拒绝的错误概率称为$\alpha$，为一类错误(type I error)。

   2. 操作特征(operating characteristic)：即零假设为假却被接受的错误概率称为$\beta$，为二类错误(type II error)。

   3. 置信水平(confidence level)：$1-\alpha$，即我们正确接受$H_0$的概率。

   4. 统计功效(statistics power)：$1-\beta$，即我们正确拒绝$H_0$的概率，简单说就是真理能被发现的可能性。我们在实验中非常重视这个指标，因为这是我们实验的目的。比如：明明零假设：胰岛素和血糖没有关系是错的，我们却接受了它，实验便失去了意义。

   5. 关键区域(critical region)：我们能够拒绝零假设的区域，记为C。![img](MIT-Data_Analysis_for_Social_Scientists(II).assets/E33A7503-C2B4-4AE4-A4BB-7AAC456C5627.jpg)

### 2.4. Power Calculation in Experimental Design

1. 在统计性实验中，我们会在实验前通过功效计算确定合适的样本数量，其中涉及的指标有：

   1. $\alpha$：显著性水平，一般选0.05。

   2. $N = N_c + N_t \quad \gamma = N_t/N$：$N$表示样本数量，$N_c$表示控制组样本数量，$N_t$表示实验组样本数量；$\gamma$表示实验组数量占比，一般为0.5。

   3. $\beta$：操作特性，对于$\beta$的选择更灵活，不过一般选择0.2，即统计功效为0.8，表示100次实验中有80次的机会成功拒绝零假设，发现真理。

   4. $\tau$：表示目标实验平均效果(target average treatment effect)，可以通过过往实验得知，也可以通过试点实验得到(pilot experiment)，或者仅仅是猜测基于扩大规模成本效益的最低实验效果(the lowest treatment effect that makes it cost effective to scale up)。因为$\tau$越大，说明估计的实验效果越好，则样本数越少。

   5. $\sigma$：对实验结果标准差的估计(an estimate of standard deviation of the outcome)。假设实验的方差和过往实验一致，可以使用过往类似实验的标准差，也可以通过试点实验获得。

2. 功效计算的相关公式：功效计算基于的假设是：实验组和控制组的标准化均值差分布满足标准正态分布。
   $$
   N = \frac{(\phi^{-1}(1-\frac{\alpha}{2}) + \phi^{-1}(\beta))^2}{\frac{\tau^2}{\sigma^2} \cdot \gamma \cdot (1-\gamma)}
   $$

![img](MIT-Data_Analysis_for_Social_Scientists(II).assets/B50E9ABE-658B-4F2A-8D50-BA951EA10DC7.jpg)

![img](MIT-Data_Analysis_for_Social_Scientists(II).assets/4C19B5B5-A579-4D58-B646-C076DDCFA8DB.jpg)

# Module 7

## 1. Causality

1. 因果关系是统计学中关注的重点问题，也是棘手难证的，比起思考什么造成了Y(eg：造成印度学校普及率低的原因)，我们更倾向于思考X的影响和缺少X的影响(eg：假设我们减少学校教室的数量，学校普及率会提高吗?)。
2. 理解因果关系的第一步要求仔细了解其反面结果(counterfactual world)，比如：研究一个项目的影响，需要知道对象不参加该项目的话，可能产生的其他行为及影响，否则，就可能错估该项目的影响(eg：领导力学前教育 vs 其他各种各样的学前教育)。
   ![img](MIT-Data_Analysis_for_Social_Scientists(II).assets/BCCDC2FF-FCE8-44BF-9615-9344B6B7F99B.jpg)
3. 因果和预测，是我们做研究的两方面目的。有时我们关注因果(eg：收紧移民政策是否影响本地人工资水平)，有时我们关注预测(eg：犯罪嫌疑人能否保释，需预测保释期间其是否会再次犯罪)，取决于我们的研究目的和感兴趣的变量。

### 1.1. SUTVA Assumption

1. 稳定单位治疗价值假设(Stable Unit Treatment Value Assumption)是很多实验的前提假设，指任何单位的潜在实验结果室是一致的，任何单位的潜在结果不会因其他单位的处理结果而变化：
   1. 实验单位之间互不干扰
   2. 排除了外部性/溢出效应(externalities or spillover)
   3. 某些实验中假设会失效，比如传染病学。
      ![img](MIT-Data_Analysis_for_Social_Scientists(II).assets/05A02ED6-C0DD-4C9F-80FD-77A5559B8E77.jpg)
      ![img](MIT-Data_Analysis_for_Social_Scientists(II).assets/4DAF6F48-F6E1-4C36-A1D4-66562EBECEAF.jpg)

### 1.2. Treatment Effect and Selection Bias

1. 实验的估计量为：实验组结果减去控制组结果，即$E(Y_i|W_i = 1) - E(Y_i|W_i = 0)$，其中可以分成两部分：
   1. 实验组效果(treatment effect on the treated)：描述了实验组的效果差异，Y(0)表示未治疗的潜在结果(实际上并未发生的反事实)，Y(1)表示实验结果。

      $E(Y_i(1)|W_i = 1) - E(Y_i(0)|W_i = 1)$

   2. 选择偏差(selection bias)：描述了实验组和控制组的未治疗的潜在结果/初始状态Y(0)的差异。比如：实验组的人可能更头痛，所以选择吃药；选择上大学的人可能拥有更好的家庭环境，所以选择上大学。

      $E(Y_i(0)|W_i = 1) - E(Y_i(0)|W_i = 0)$

      ![img](MIT-Data_Analysis_for_Social_Scientists(II).assets/958E1E16-6E7A-4D33-BABE-C4DA8458396B.jpg)

2. 选择偏差问题是我们在实验过程中需要避免的，这样才能得到实验的真正效果。亦就是说，我们要保证实验组和控制组的初始状况是一致的。解决这一问题的方法是：**随机化选择实验对象**。

### 1.3. Type of RCT

1. 随机化的方法有：
   1. 完全随机(completely randomized)
   2. 分层随机(stratified randomized)
   3. 成对随机(pairwise randomized)：以成对为对象进行随机化，成对随机是分层随机的一种特殊情况，其中每个层均包含两个单位，一个属于实验组，一个属于控制组。
   4. 聚类随机(clustered randomized)：不以个体为单位，以组为单位进行随机(eg：一组学校、一片区域)。
      ![img](MIT-Data_Analysis_for_Social_Scientists(II).assets/23CDD314-FA78-4EF6-A6D4-86888DE1CF47.jpg)
      ![img](MIT-Data_Analysis_for_Social_Scientists(II).assets/ADD30E7E-0FD4-4E89-8FA6-5670128A4DEE.jpg)

## 2. Analyzing Randomized Experiments

1. 随机对照试验(Randomized Control Trials, RCT)社会科学实验中重要的一种方式，通过设置对照组和实验组，对比两组别的效果差异得出有意义的结论。**我们一般选择样本均值作为效果的估计量(sample mean = outcome)。

### 2.1. standard Neyman analysis

1. 在随机对照试验中，试验组的均值减去对照组的均值就是估计的平均试验效果(average treatment effect)，它作为一个无偏估计量被广泛应用于估计：
   $$
   \widehat{\tau} = E(Y^{obs}[W_i =1] - E(Y^{obs}[W_i =0]
   $$

![img](MIT-Data_Analysis_for_Social_Scientists(II).assets/380A6238-689A-4527-836D-801C60D0CDEF.jpg)

2. 平均试验效果的估计方差为实验组均值估计方差和对照组均值估计方差的和（假设试验组和对照组相互独立，即试验随机分配(treatment is randomly assigned)），即：
   $$
   V_{Neyman}(\widehat{\tau}) = \frac{S_c^2}{N_c} + \frac{S_t^2}{N_t}
   $$

3. 平均试验效果估计量的置信区间：
   $$
   CI = [\widehat{\tau} + Z/t * \sqrt{V(\widehat{\tau})}, \widehat{\tau} - Z/t * \sqrt{V(\widehat{\tau})]}
   $$

![img](MIT-Data_Analysis_for_Social_Scientists(II).assets/780B65F3-15A1-4A46-8F8C-70FF3D14F0F8.jpg)

4. t检验(Neyman hypothesis test)：$t = \frac{\widehat{\tau}}{\sqrt{V}}$，当t足够大(eg：t>1.96)时，可以拒绝零假设(表达：I can reject the hypothesis that the treatmenteffect on medical expenditure is 0 at the 5% level.)。当t太小或CI包含0时，则不能拒绝零假设。
   ![img](MIT-Data_Analysis_for_Social_Scientists(II).assets/31B4D552-131D-4ECC-A776-0F8094C46344.jpg)

5. 如果零假设不是平均试验效果为0，而是其他数字，如零假设：平均试验效果为0.5，则可以通过以下两种方式检验：
   1. CI是否包含该数字0.5，若是包含，则不能拒绝。
   2. 计算该数字的t统计量，即$t = \frac{\widehat{\tau}-0.5}{\sqrt{V}}$

### 2.2. Fisher Exact Test

1. 费雪精确检验和标准经典检验(standard classical statistical measure)的世界观完全不同。**费雪精确检验将RCT的数据集看成一个总体(不是样本vs总体)，检验的是sharp null，即是否每个对象的实验效果都不等于0(the treatment effect is 0 for everybody.)**。

2. 虽然费雪精确检验把数据看成总体，但其还是存在不确定性。其**不确定性在于结果的随机性，即每个实验对象都有两个潜在的结果(实验组vs对照组)，但我们实际上只能观察其中一个，最终只能得到一半的数据**。那么，观察到的数据的多种排列组合导致了总体数据的不确定性。

3. 在今天，随着大数据的发展，费雪精确检验越来越被广泛应用。对于某些大公司，比如facebook、twitter的数据库等可以直接看做一个总体进行实验研究。

4. 费雪试验的步骤：

   1. 假设每个对象的实验效果为0，则可以得到反事实，补充另一半未能观察到的数据。
      ![img](MIT-Data_Analysis_for_Social_Scientists(II).assets/6D58B3D2-C4FD-4773-91D8-4A0FBD69C3BA.jpg)

   2. 计算实验效果$\widehat{\tau} = E(W_i=1)-E(W_i=0)$

   3. 列出对象实验组和对照组的所有组合。如假设实验组数量为3，对照组数量为3，则一共有$C_6^3 = 20$种组合方式(permutation test)。
      ![img](MIT-Data_Analysis_for_Social_Scientists(II).assets/45D5714B-FC2D-4DF9-9501-FD2D7EE21DEC.jpg)

   4. 计算所有组合的假实验效果Y，Y就是零假设下能得到的所有可能结果。当数量太大时，可以选择K个进行随机计算。
      ![img](MIT-Data_Analysis_for_Social_Scientists(II).assets/ECDE1AAF-9D36-4AE2-A290-1FB98FC90FFD.jpg)

   5. 计算比观察到的实验效果$\widehat{\tau}$大或相等的Y绝对值的个数(包含实际实验效果本身)，计算P值。
      $$
      P = \frac{n(|Y|≥\widehat{\tau})}{n_{permutation}}
      $$

   6. 当P小于0.05时，说明在零假设下，所有的组合只有不到5%的可能性等于或好于观察到的实验效果。也就是说，在5%的显著性水平下，我们可以拒绝零假设。

5. 费雪精确检验的代码：

```
# 1. 载入相关的包
library(perm)
library(modelr)
library(np)
library(tidyverse)

# 2. 创建组合
perms <- chooseMatrix(8,4) # 实验组4，对照组4

# 3. 创建实验结果向量
A <- matrix(c(0.462, 0.731, 0.571, 0.923, 0.333, 0.750, 0.893, 0.692), nrow=8, ncol=1, byrow=TRUE)

# 4. 计算所有组合的fake实验效果
treatment_avg <- (1/4)*perms%*%A
control_avg <- (1/4)*(1-perms)%*%A
test_statistic <- abs(treatment_avg-control_avg)

# 5. 找到观察到的实验效果
rownumber <- apply(apply(perms, 1, 
                            function(x) (x == c(0, 1, 0, 0, 0, 1, 1, 1))), # 实验组和对照组编号
                            2, sum)
observed_test <- test_statistic[rownumber == 8]

# 6. 计算大于/等于实际实验效果的Y
larger_than_observed <- (abs(test_statistic) >= observed_test)
count_of_larger <- sum(larger_than_observed)

# 7. 计算P值
count_of_larger/length(test_statistic)
```

## 3. More Exploratory Data Analysis

1. 在此之前，我们进行过单变量的探索性数据分析，包括直方图、CDF图、核密度估计图。接下来我们通过非参数比较和回归(Nonparametric Comparisons and Regressions)来进一步进行探索多变量之间的关系。

[探索性数据分析](https://learning.edx.org/course/course-v1:MITx+14.310x+2T2020/block-v1:MITx+14.310x+2T2020+type@sequential+block@8dff8a1c24d04d08ad7f93b79a958f48/block-v1:MITx+14.310x+2T2020+type@vertical+block@a81d21a3b70d4b13b18b6b99631fd79a)

### 3.1. Comparing Distributions

1. CDF图里的一阶随机占优定义：

[first order stochastic](https://learning.edx.org/course/course-v1:MITx+14.310x+2T2020/block-v1:MITx+14.310x+2T2020+type@sequential+block@8dff8a1c24d04d08ad7f93b79a958f48/block-v1:MITx+14.310x+2T2020+type@vertical+block@be649c33cb0d438f9c7052f192ae2c29)

1. KS检验(The Kolmogrov Smirnov Test)可用于比较两个变量的分布是否相似，如检验RCT的实验组和对照组分布是否相同。

   1. 零假设：CDF(X) = CDF(Y)

   2. 计算CDF的最大垂直距离$D_{nm}$

   3. 不管X和Y的分布是什么，当n和m趋于无穷时(n表示X的数量，m表示Y的数量)，$(\frac{nm}{n+m})^{\frac{1}{2}} D_{nm}$满足ks分布。当$D_{nm}$大于特定的阈值$C(\alpha)(\frac{nm}{n+m})$(阈值基于分布的形状)时，我们可以拒绝零假设，认为X和Y的分布存在差异。

   4. 我们也可以使用变量X和特定分布的CDF进行ks检验，以判断X的分布。eg：X和具有相同均值和标准差的正态分布、Y和指数分布。

   5. 当然，因为ks检验只是取了一个点的距离作为统计量，它的统计功效不足，可能会产生很多二类错误(type two error)。也就是说，零假设是错误的，但我们却没法拒绝它。

```
# ks检验——X vs Y
ks.test(X, Y)

# ks检验——X vs 正态分布
ks.test(X, "pnorm", mean=mean(X), sd=sd(X))
```

### 3.2. Nonparametric Regression

1. 非参数回归是指在不计算函数g(x)的前提下，探索变量X和Y的关系：
   $$
   y = g(x) + \epsilon
   $$

[scatter plot](https://learning.edx.org/course/course-v1:MITx+14.310x+2T2020/block-v1:MITx+14.310x+2T2020+type@sequential+block@8dff8a1c24d04d08ad7f93b79a958f48/block-v1:MITx+14.310x+2T2020+type@vertical+block@33b6d790139a4a62824ce1b1e7d9ea10)

2. 核回归和核密度估计有点类似，内核(kernel)计算每一个区间x的y的加权回归均值(the weighted average of the yin the neighborhood of the x)。需要注意的是：
   1. 内核的选择并不重要，任何对称的钟形内核都可用，如正态内核、Epanechnikov内核等 (it's not very important of which kernel to use)。
   2. 带宽(bandwidth)会影响核回归曲线。带宽越大，曲线越光滑。R会默认选择MSE最小的带宽，不需要我们自行设置。
   3. ggplot包不包含核回归(kernel regression)，载入np包可使用。

[kernel regression](https://learning.edx.org/course/course-v1:MITx+14.310x+2T2020/block-v1:MITx+14.310x+2T2020+type@sequential+block@8dff8a1c24d04d08ad7f93b79a958f48/block-v1:MITx+14.310x+2T2020+type@vertical+block@500476704fe44a338af232606d92841e)

3. 加权回归(wighted regression)：比起加权平均，人们认为加权回归的表现更好，即不是计算一段区间内y的加权平均值，而是计算$y =ax +b$的加权预测值。加权回归的好处主要体现在：
   1. 在边缘处的拟合更好(fits better at the edges)，因为它是进行预测，因此，当数据出现断层时，它能根据预测拟合而不会发生偏移。
   2. 实际上加权回归也可以看成一个内核，即广义上的核回归。

# Module 8

## 1. Bivariate Linear Model

> 线性模型被广泛用于研究多个变量之间的联合分布/条件分布(joint distribution/conditional distribution)并估计它们的参数。我们估计参数的方法称为线性回归(linear regression)。

1. 我们为什么关心联合分布及其参数呢？
   * 预测(prediction)：尿布和啤酒
   * 确定因果关系(determining causality)
   * 更好地了解世界(just understanding the world better)：购车当天的天气会影响人们对车辆的选择，晴朗的日子倾向于购买敞篷车，而雨天倾向于购买轿车。
2. 首先我们来看一下简单的二元线性回归。

### 1.1. Model Fomula

* 二元线性回归模型可表示为：
  $$
  Y_i = \beta_0 +\beta_1 x_i +\epsilon_i, \qquad i = 1,..., n
  $$

各符号分别表示：

* $Y_i$：因变量/被解释变量/回归变数(dependent variable/explained variable/regressand)

* $x_i$：自变量/解释变量/回归量(independent variable/explanatory variable/regressor)

* $\epsilon_i$：误差(error/unobserverd random varible)

* $\beta_0, \beta_1$：被估计的参数/回归系数(parameters to be estimated/regression coefficients)![img](MIT-Data_Analysis_for_Social_Scientists(II).assets/33391DA4-86AC-42F6-887E-07DA1D6D008D.jpg)

* 经典线性回归模型基于以下假设：

1. $x_i$和$\epsilon_i$不相关(uncorrelated)
2. x是变化可识别的(identification)：$\frac{1}{n} \sum (x_i - \overline{x})^2 >0$
   1. 即排除一种情况：x固定等于一个值
   2.  为了得到$Y_i$，x必须是有变化的
3. 误差均值为0：$E(\epsilon_i) = 0$
   1. 我们没办法区分误差均值是否等于0，因此假设误差均值为0。![img](MIT-Data_Analysis_for_Social_Scientists(II).assets/02B49D38-1603-4E82-AEFD-F30C0D5E1961.jpg)
4. 方差齐性(homoscedasticity)：$E({\epsilon_i}^2) = \sigma^2$
   1. 在实际工作中，我们经常使用残差图(residual)判断是否方差齐性(误差为纵轴，自变量为横轴），在这里，我们先假设方差齐性。
5. 无序列相关(no serial correlation)：$E({\epsilon_i}{\epsilon_j}) = 0 \qquad \text{注：i不等于j}$
   1. 也就是说，误差之间是不相关的。![img](MIT-Data_Analysis_for_Social_Scientists(II).assets/A1F56430-0043-442B-A115-64E8A01A1FD1.jpg)
      除了以上假设，有时还会增加以下两个假设：
6. x在重复样本中是固定的，即非随机性(x are fixed in repeated samplesor non-stochastic.)。
7. 将假设3-5结合起来，则可以假设$\epsilon_i$满足独立同分布的正态分布$N(0,\sigma^2)$。

* 模型属性：

  1. $Y_i$的期望值为$\beta_0 +\beta_1 x_i$

  2. $Y_i$的方差等于误差方差

  3. $Y_i$的协方差为0

  ![img](MIT-Data_Analysis_for_Social_Scientists(II).assets/B8F93E2D-F355-4D40-AE83-61F0D332A97D.jpg)

### 1.2. Find Parameters

* 课堂介绍了三种方法找到最优的参数：最小二乘法(least squares)、反向二乘法(reverse least squares)、最小绝对偏差法(least absolute deviations)。
  ![img](MIT-Data_Analysis_for_Social_Scientists(II).assets/47E641E7-F394-479E-BCB0-CA3053068B0A.jpg)

1. 最小二乘法是最常用的方法，也被称为普通二乘法(ordinary least squares, OLS)。
   * 在经典线性回归模型下，利用最大似然估计法MLE，OLS对参数进行了无偏估计且方差最小、具有一致性，因此，比起其他方差，它是最有效的(most efficient)。
   * 我们通过推导公式直接得到近似的参数(无须进行最小化)。
     ![img](MIT-Data_Analysis_for_Social_Scientists(II).assets/AC02E964-BC7A-4858-8318-565E3E6A1C3E.jpg)
2. 反向二乘法：反向计算X的偏差
   ![img](MIT-Data_Analysis_for_Social_Scientists(II).assets/DD1E27A8-A06D-4117-BCFC-7019C0342C70.jpg)

### 1.3. Some Estimators and their distribution

* 残差和预测值(residual and prediction)：$\hat{\epsilon} = y_i - \hat y_i$

* 估计参数：$\hat{\beta_n}$：

  * $\sigma^2$(误差方差)越大，$\hat{\beta_n}$的方差越大

  * $\sigma_x^2$(自变量方差)越大，$\hat{\beta_n}$的方差越小

  * 样本量n越大，$\hat{\beta_n}$的方差越小

  * 如果样本均值大于0，则$\hat{\beta_n}$的协方差为负，即对$\hat{\beta_0}$的低估会导致对$\hat{\beta_0}$的高估。![img](MIT-Data_Analysis_for_Social_Scientists(II).assets/10F690A8-6A55-4E9C-8E74-9E42160617F9.jpg)

* 估计参数的分布：

  * 因为$\hat{\beta_n}$是误差方差的函数，误差方差满足iid的正态分布，因此，估计参数的分布也满足正态分布。

  * 总体误差方差是不可知的，因此估计误差方差的无偏估计为：
    $$
    \hat{\sigma^2} = \frac{1}{n-2}\sum \hat{\epsilon^2}
    $$

### 1.4. goodness-of-fit

* SSR、SST和SSM：

  * 残差平方和(sum of squared residuals,SSR)：残差平方和可用于衡量模型的拟合度，但其大小受单位影响(it is not unit free)，不方便判断。
    $$
    SSR = \sum(Y_i - \hat{Y_i})^2 = \sum(\hat{\epsilon})^2
    $$

  * 总平方和(total sum squares,SST)：为了消除单位的影响，我们引入SST，SSR/SST不受单位影响。SST引入了y的样本均值$\overline{Y}$：
    $$
    SST = \sum(Y_i - \overline{Y})^2
    $$

  * SSR/SST满足：
    $$
    0<= SSR/SST <=1\\
    \text{注：SSR即回归线误差，是我们能得到的最小误差(least square error)}
    $$

  * SSM：模型平方和，计算回归线上的预测值$\hat{Y_i}$和样本均值$\overline{Y}$之间的差值。
    $$
    SSM = \sum (\hat{Y_i}-\overline{Y})^2\\
    SST = SSR +SSM
    $$
    ![img](MIT-Data_Analysis_for_Social_Scientists(II).assets/0D6D94BB-4D31-43A8-877B-15DAA06BBE0E.jpg)

------

* $R^2$和$adjust-R^2$

  * $R^2$：$R^2$被广泛用于衡量模型拟合度，$R^2$越大，说明模型拟合度越好。
    $$
    R^2 = 1- SSR/SST
    $$

  * 除了衡量拟合度外，$R^2$还可用于假设检验零假设$\beta_1 = \beta_2 = ... =\beta_k = 0$(假设有k个系数)。当$\frac{(n-k)R^2}{(k-1)(1-R^2)}$(零假设下满足F分布)足够大时，我们可以拒绝零假设。![img](MIT-Data_Analysis_for_Social_Scientists(II).assets/D9E57A86-5806-4B85-8B15-D52884502BDF.jpg)

### 1.5. Interpret estimates

* $\hat{\beta_n}$表示了X每1单位的变化对Y的估计影响。

## 2. Multivariate Linear Model

> 实际工作中，我们用得更多的是多元线性回归模型，因为一个因变量可能受多个自变量影响。

### 2.1. Model Formula

* 多元回归模型可表示为：
  $$
  Y_i = \beta_0 + \beta_1 X_{1i} + \beta_2 X_{2i} +...+ \beta_k X_{ki} + \epsilon_i \qquad i = 1,...,n
  $$

* 使用矩阵求和方式，可以将公式简化为：
  ![img](MIT-Data_Analysis_for_Social_Scientists(II).assets/CEFB533B-B725-46BB-B35A-A05699921612.jpg)
* 基于多元线性回归，我们可以将假设压缩为以下两个：
  1. 可识别性(identification)：n>k+1；矩阵X满秩(k+1，即回归量之间线性独立，且$X_TX$可逆)。
  2. 误差分布(error distribution)：$E(\epsilon) = 0, E(\epsilon \epsilon^T) = Cov(E) = \frac{\sigma^2}{n}$,即误差满足$N(0,\frac{\sigma^2}{n})$的正态分布。
     ![img](MIT-Data_Analysis_for_Social_Scientists(II).assets/B0126DA2-98D3-44FA-BEDD-017A800EC716.jpg)
     （注：误差之间相互独立，因此不同误差的协方差为0.）

### 2.2. Multicollinearity

* 多重共线性是多元线性回归中需要重点关注的问题。
  * 当某几个自变量之间存在完美共线性时，需要剔除相应变量，不然就无法识别这些变量之间的变化(no enough variation to identify the effects of these varibles)
  * 当某几个变量之间存在近似共线性关系时，我们仍可以估计模型，但结果存在较大的方差。剔除变量，则需要容忍较大的偏差。
* 哑变量的多重共线性陷阱：对于分类变量，我们创建哑变量时，可以有以下操作：
  * 删除其中一个分类的哑变量，以消除共线性问题。
  * 或者，不估计截距参数(not estimate an intercept)。

### 2.3. Deriving Estimators(OLS)

* 估计参数$\hat{\beta}$的目的是什么呢？——使误差最小化。即：
  $$
  \hat{\epsilon^T} \hat{\epsilon} = (Y-X \hat{\beta})^T (Y-X \hat{\beta}) \quad \text{最小化}
  $$

* 因此对函数进行求导且令导数等于0，则得到$-2 X^T (Y-X \hat{\beta}) =0$,则可得到：
  $$
  X^T Y = X^T X \hat{\beta}\\
  \hat{\beta} = (X^T X)^{-1} X^T Y
  $$

### 2.4. Estimators' Distribution

![img](MIT-Data_Analysis_for_Social_Scientists(II).assets/C0C31CF6-F191-4B06-AF33-2DC9C7438D19.jpg)

* 其中，$\hat{\sigma^2}$表示估计的误差方差/因变量方差(residual variance estimator)，$\hat{\epsilon^T}\hat{\epsilon}$表示残差(residual)。也就是说，方差受残差和自由度(样本量n-回归量个数k)的影响。

### 2.5. hypotheses & inference

![img](MIT-Data_Analysis_for_Social_Scientists(II).assets/ECF7265B-5DE0-4F64-8C4D-A489F20EDCFD.jpg)

* $\vec{R}$是一个限制向量，用于检验$\beta$。其行数为我们想要检验的限制条件$r$(restrictions)，列数为模型的参数数量$k+1$。

  1. 假设$\vec{R} = \{0\,1\, 0\,...\,0\}$，$\vec{c} = \{0\}$，则对应的零假设为：$\beta_1 = 0$。

  2. 假设$\vec{R} = \{0\,0\, 1\,...\,0\}$，$\vec{c} = \{5\}$，则对应的零假设为：$\beta_2 = 5$

  3. 假设$\vec{R} = \{0\,1\, -1\,...\,0\}$，$\vec{c} = \{0\}$，则对应的零假设为：$\beta_1 = \beta_2$

* 该类型假设满足F分布，即$T \sim F_{r,n-(k+1)}$，当F足够大时，我们拒绝零假设。

  ![img](MIT-Data_Analysis_for_Social_Scientists(II).assets/8B8C5C3D-8899-40F9-9BB2-3369C7AD9603.jpg)

* 如何在R中操作该假设检验：

```
library(car)
R <- c(0,1,0)
linearHypothesis(model, R)
```

### NOTE: Cov & Var

1. 协方差（Covariance）用于衡量两个随机变量的联合变化程度。而方差是协方差的一种特殊情况，即变量与自身的协方差。

2. 期望值分别为$E(X) = \mu$和$E(Y) = \nu$的两个具有有限二阶矩的实数随机变量X与Y之间的协方差定义为:
   $$
   cov(X,Y)= \frac{1}{n} \sum ((X- \mu)(Y- \nu)) = E((X- \mu)(Y- \nu)) = E(X \cdot Y) - \mu \nu
   $$

3. 协方差具有以下特点和属性：
   1. 如果X 与Y 是统计独立的，那么二者之间的协方差就是0，因为：

      $E(X\cdot Y)=E(X)\cdot E(Y)=\mu \nu$

   2. $cov(X, X) = Var(X)$

      $cov(X, Y) = cov(Y, X)$

      $cov(aX,bY) = ab \, cov(X, Y)$

   3. 协方差矩阵：

      $cov(X,Y)= E((X- \mu)(Y- \nu)^T)$

      $cov(X,Y)^T = cov(Y,X)$

4. 方差是特殊的协方差，描述一个随机变量的离散程度。设X为服从分布F的随机变量，如果$E(X)$是随机变数X的期望值（平均数$\mu =E(X)$），则随机变量X或者分布F的方差为：
   $$
   Var(X) = E((X- \mu)^2) = Cov(X, X)
   $$

5. 方差具有以下特点和属性：
   1. $Var(X) = E(X^2-2XE(X)+(E(X)^2) = E(X^2)-2E(X)E(X)+(E(X))^2 = E(X^2)-2(E(X))^2+(E(X))^2 = E(X^2)-(E(X))^2$
   2. $Var(X+a) = Var(X)$
   3. $Var(aX) = a^2Var(X)$
   4. $Var(aX+bY) = a^2Var(X) + b^2Var(Y) + 2ab \, Cov(X,Y)$

# Module 9

## 1. Practical Issues with Regressions

### 1.1. Dummy Variables

1. 定义：哑变量(dummy variables)也称为指示变量(indicator variables)，取值仅为0或1，举例：

   1. RCT：实验组为1，对照组为0
   2. 性别：1表示男性，0表示女性

2. 当线性回归中，只有一个自变量且为哑变量时，即：$Y_i = \beta_0 + \beta_1 D + \epsilon_i$，则：

   * $\hat{\beta} = \overline{Y_{D=1}}-\overline{Y_{D=0}}$
   * 该最小二乘法的回归模型有时被用于RCT中比较实验组和对照组的差异，不过OLS标准误计算方法比起尼曼标准误，存在一定缺陷，OLS对整个样本进行计算，忽略了实验组和对照组之间的比例关系(当n足够大/$n_C = n_T$时，两者无太大差别)。

3. 分类变量转换成哑变量( from a categorical variable to dummy variables)：

   1. 分类变量的值是无意义的(uninterpretable)，并没有描述变量的任何内容，这就是为什么应该将分类变量编码为哑变量的原因。

   2. 为避免共线性，开始回归之前，需要删除其中一个哑变量，留下的哑变量参数可以解释为：**该组相对被删除组对因变量影响的差别**。

   3. 哑变量交互作用(interaction)：假设回归模型中有两个哑变量，1个是RCT组别$R$，1个是性别$G$，则模型为：
      $$
      Y_i =  \beta_0 + \beta_1 R + \beta_2 G + \beta_3 R \cdot G + \epsilon_i
      $$

      * $\beta_0$：控制组中女性的均值(令其他项=0)

      * $\beta_1$：女性中，实验组相对于对照组的区别

      * $\beta_2$：控制组中，男性相对于女性的区别

      * $\beta_3$：实验组中，男性相对于女性的区别

      * 这个包含两个哑变量的模型在实证经济研究中也被叫做Difference-in-Differences Model，经常被用于研究颁布某项法律的差异影响(differential effects)。

### 1.2. Other Functional Form Issues

1. 转换自变量和因变量：假设自变量和因变量存在非线性关系$Y_i =AX_2^{\beta_1}X_2^{\beta_2} \epsilon$， 则可以进行log变换：
   $$
   log(Y_i) = \beta_0 + \beta_1 logX_1 +\epsilon_I
   $$

   * 其中$\beta_1$也被称为弹性(elasticities)，表示$X_1$变化1%，$Y_i$变化$\beta_1\%$。

2. Box COx Transformation：
   $$
   \frac{1}{Y_i} = \beta_0 + \beta_1 X_1 +\epsilon_I
   $$

3. 自变量的非线性转换：
   1. 多项式(polynomial)：也叫做级数回归(series polynomial)，与核回归(kernel regression)具有异曲同工之妙。
      ![img](MIT-Data_Analysis_for_Social_Scientists(II).assets/11F08767-FDAE-469B-A7B7-C4FF6C8D2035.jpg)
   2. 其他方式：log转换、多变量交互、分类变量/哑变量转换(dummy variable approximation)
      ![img](MIT-Data_Analysis_for_Social_Scientists(II).assets/079DD79A-E047-4AEE-AF1D-97DA95ABAB3B.jpg)

4. 局部线性回归(local linear regression)：本质上就是核回归，只不过比起核回归用内核计算局部加权平均值，LIR用线性回归模型预测每一个小区间。其优点有：
   1. 在边界的估计上偏差更小(less bias at the boundaries)
   2. 模型的斜率是我们感兴趣的( the slope is often of interest)
      ![img](MIT-Data_Analysis_for_Social_Scientists(II).assets/DC279259-CA9D-491E-8DAE-A9B4DE86E03A.jpg)

### 1.3. Regression Discontinuity Design

1. 关于不连续的回归模型(RDD)设计，需要注意边界值预测的准确性，警惕非线性关系伪装成不连续性。

   1. 模型设计举例：法律规定驾车年龄为18岁，记为：$D_a = 1\, \text{if} \, a>=18$，研究其与机动车事故致死率的关系，则我们可以假设致死率在18岁会有一个跃升。
   2. 警惕非线性关系：如何解决？
      1. 参数方法：利用多项式对整个模型或边界两侧进行验证。
         ![img](MIT-Data_Analysis_for_Social_Scientists(II).assets/D2094B47-DA14-4455-88A4-19DA8992441D.jpg)
      2. 非参数方法：将带宽减少到不连续点的附近。

2. 相关可以设计的模型有：

   1. 简单分析：增加一个哑变量来移动截距：
      $$
      Y_i = \beta_0 +\beta_1 D_{a} +\beta_2 a + \epsilon
      $$
      ![img](MIT-Data_Analysis_for_Social_Scientists(II).assets/DE358767-C0A3-42B2-9A12-3971A4C489E1.jpg)

### 1.4. Omitted Variable Bias

1. 构建模型遗漏的变量会导致模型偏差(omittedb variable bias)。举例：假设你要研究孩子上私立学校和公立学校对他们未来收入的影响，因为孩子上私立的学校的前提还可能有：SAT成绩高、父母收入高等潜在因素，这两个也可能会影响他们的未来收入，因此构建模型时，需要将这两个变量也考虑进去，否则模型就缺乏说服力。
   ![img](MIT-Data_Analysis_for_Social_Scientists(II).assets/015DA8BB-B9D6-493E-BE7F-FF221A60B044.jpg)

2. OVB公式：假设正确的模型包括我们要研究的变量$X_1$和对结果产生影响的$X_2$，但我们无法获得有关$X_2$的数据，则需要构建评估模型和辅助回归模型(ancillary regression)：
   ![img](MIT-Data_Analysis_for_Social_Scientists(II).assets/3590DD9D-68A1-471A-B062-C6B256D2C267.jpg)

   1. $\text{正确模型：} Y_i = \beta_0 + \beta_1 X_1 + \beta_2 X_2 + \epsilon_i$

   2. $\text{评估模型：} Y_i = \alpha_0 + \alpha_1 X_1 + \epsilon_i$

   3. $\text{辅助回归模型：} X_{2i} = \delta_0 + \delta_1 X_1 + \epsilon_i$

   4. $\text{OVB为：} OVB = \alpha_1 - \beta_1 = \delta_1 \beta_2$

      ![img](MIT-Data_Analysis_for_Social_Scientists(II).assets/89C7D4D5-6F90-41D6-A5DC-722BB5E1F8D2.jpg)

3. 更加高级的OVB方法：

   1. 匹配方法(matching methods)：先将所有可能有关的变量构建一个负责的回归模型，然后进行倾向匹配。
   2. 机器学习方法(machine learning)：对X和Y进行拉索回归，并构建新模型。
      ![img](MIT-Data_Analysis_for_Social_Scientists(II).assets/23F943C8-5F8F-407F-B069-6EA43D43CC0C.jpg)

# Module 10

## 1. Endogeneity and Instrumental Variables

1. 当我们有合理的理由怀疑因变量影响了自变量的时候，就产生了内生性问题(endogeneity problem)，因为模型系数就包括了自变量对因变量的影响以及因变量对自变量的影响。
2. 关于回归模型的因果解释，取决于我们进行的一系列假设的可靠性(credibility)，与模型设计(eg:简单/DID/RDD)无关。不管是多复杂的模型设计，如果自变量和因变量可能是相互的关系(mutual relationship)，都会导致内生性问题。
3. 内生性问题举例：
   1. 健康和锻炼：身体健康的人会更倾向于锻炼，而锻炼也有利于身体健康。
   2. 政府崩溃和危机：国家危机可能会导致政府崩溃，而政府崩溃也可能会造成国家危机。
   3. 专业和练习：对于某个技能，练习让你更专业，更专业的你会更多的练习。

### 1.1. Endogeneity vs OVB

1. 内生性问题和遗漏变量偏差的主要区别在于：一个是自变量和因变量的相互影响问题，一个是遗漏了因变量的影响因素从而导致感兴趣的自变量系数产生偏差(upward biased)。
   ![img](MIT-Data_Analysis_for_Social_Scientists(II).assets/A4C9BA43-4A18-4EE6-AB55-1E97C6274629.jpg)

### 1.2. Assigning "Instrument"

1. 以研究教育的影响为例(correlation between education and many outcomes)，因为教育是自由且基于个人或家庭意志的，和诸多方面都存在相互的关系。该研究困难在于：
   1. 教育是自由的，我们没办法直接对教育进行变量分析
   2. 随意选择一个影响教育的变量可能会产生内生性问题。
      ![img](MIT-Data_Analysis_for_Social_Scientists(II).assets/38062A92-525F-4721-9543-CAEA363CD274.jpg)
2. 因此，为解决这一问题，方法之一是引入工具变量。利用工具变量探索其对教育的影响，假设该工具变量对因变量(如收入)没有直接影响，则可以推断出教育对收入的影响，这就叫**工具变量法**（method of instrumental variables,简称IV)
   ![img](MIT-Data_Analysis_for_Social_Scientists(II).assets/05D2CBB3-CE8D-4B63-AFF4-7E6F9B8AF6F0.jpg)
3. 以教育为例，可以考虑的工具变量有：
   1. 奖学金：利用奖学金刺激部分学生接受更多的教育。
   2. 教育免费：对部分学生提供教育免费，鼓励进一步学习
   3. 课后辅导：开展课后辅导项目，让部分学生接受更多的教育
4. 如何进行效果的转化呢？
   ![img](MIT-Data_Analysis_for_Social_Scientists(II).assets/F0897E7C-1041-4806-AFC2-625DA31A503C.jpg)
   ![img](MIT-Data_Analysis_for_Social_Scientists(II).assets/EE001CAF-8B3A-4ACB-8C3F-313896FF9B9A.jpg)

* 其中$E(\epsilon_i|Z_i= 1) - E(\epsilon_i|Z_i= 0) = 0$基于以下假设：

  1. Z的样本是随机的(randomly assigned)，且Z对Y的影响被很好地估计(the effect of Z on Y can be estimated well)。

  2. 工具变量Z对因变量Y没有直接的影响，即Z和Y没有直接的因果关系(no direct effect/exclusion restriction)。

  3. 工具变量Z对自变量A的影响是显著的。

  ![img](MIT-Data_Analysis_for_Social_Scientists(II).assets/F545D2BD-EC67-42BA-A82C-50FE3C01C52A.jpg)

* 以上3个假设，只要有一点点违反就会造成巨大的偏差，因此，需要慎重选择IV。

### 1.3. The Wald Estimate

1. 沃尔德估计是工具变量估计量的最简单形式，通过工具变量$Z_i$估计了$A_i$对$Y_i$的影响：
   $$
   \hat{\beta} = \frac{E(Y_i| Z_i = 1)-E(Y_i| Z_i = 0)}{E(A_i| Z_i = 1)-E(A_i| Z_i = 0)}
   $$

* 分母：第一层关系(first stage relationship)
* 分子：简化形式关系(reduced form relationship)
  ![img](MIT-Data_Analysis_for_Social_Scientists(II).assets/048DD5B6-394F-4C78-9BDA-4ACA4E6DEAEC.jpg)

### 1.4. Local Average Treatment Effect

1. 实际上我们知道因果关系的系数不一定是常数，举例：教育对分数的影响可能是因人而异的。我们利用沃尔德估计得到只是局部平均治疗效果(LATE),即那些被要求实验的遵守规则者(complier)的效果(the effect of the treatment on those who are compelled by the instrument to get treated)
2. 因此，在做IV时，我们需要注意，遵守规则者所在的群体是否是我们感兴趣的主体。
   ![img](MIT-Data_Analysis_for_Social_Scientists(II).assets/CBBC3945-B043-4F09-AD26-BF1B16A39A3C.jpg)

### 1.5. Two Stage Least Squares

1. 除了计算均值差及比值，我们也可以使用回归框架得到沃尔德估计，其优点是适用性更广，工具变量不是哑变量也同样适用。

   1. 第一层回归：$A_i = \nu_0 + \nu_1 Z_i + \epsilon_i$

   2. 简化形式回归：$Y_i = \gamma_0 + \gamma_1 Z_i + \epsilon_i$

   3. 计算第一层回归，得到$\hat{A_i}$
   4. 整理1和2的回归，得到最终模型：$Y_i = \alpha + \beta \hat{A_i} + \epsilon_i$

   5. $\hat{\beta}$就是沃尔德估计。

2. 利用两步最小二乘法(two stage least squares, 2SLS)计算，可以得到：
   $$
   \hat{\beta} = \frac{Cov(Y_i, \hat{A_i})}{Var(\hat{A_i})} = \frac{Cov(Y_i, \nu_0 + \nu_1 Z_i)}{Var(\nu_0 + \nu_1 Z_i)} = \frac{\nu_1Cov(Y_i,Z_i)}{\nu_1^2 Var(Z_i)} = \frac{\gamma_1}{\nu_1}
   $$

3. 假设我们不止一个工具变量，而是一个工具变量矩阵，即假设我们感兴趣的变量是$X_1$,则IV矩阵为$Z = \{Z_1,...,Z_k,X_2\}$，其中$X_2$作为控制变量(control variable)也被包含在矩阵中，如性别、年龄，因为工具变量有且达到了控制条件才有可能生效。则：

   1. 第一步：$X_1 = \nu_0 + \nu_1 Z_1 + ... + \nu_k Z_k + X_2 + \epsilon_i$

   2. 第二步：$Y_i = \alpha + \beta \hat{X_1} + X_2 + \epsilon_i$

      ![img](MIT-Data_Analysis_for_Social_Scientists(II).assets/29311DD6-A5C4-42D6-837B-6B415DFE041F.jpg)

   3. 利用矩阵运算,假设Z矩阵和X的变量数量相同，则为：=(*Z**X*)*Z**Y*

   4. 矩阵越大，方差就越大，公式为：*V**a**r*(*I**V*)=*σ*(*Z**X*)*Z**Z*(*Z**X*)

4. 在R中运行IV：ivreg()

```
library("AER")

iv <- ivreg(formula = total_score ~ shs_complete + region.f | treatment + region.f, data = data)

summary(iv)
```

![img](MIT-Data_Analysis_for_Social_Scientists(II).assets/1966A69A-772D-4641-B909-0C4E99CAF103.jpg)

## 2. Experimental Design

1. 实验设计需要注意的主要问题有：1.随机化

### 2.1. Randomization

1. 随机化是指对实验对象/实验群体随机进行干扰。随机化的单位是功效(power)。
   ![img](MIT-Data_Analysis_for_Social_Scientists(II).assets/B5299FC0-C5EC-4EB2-BB47-24CA38B9AAB8.jpg)
2. 随机化包括简单随机、分层随机和聚类随机等三种方法：
   1. 简单随机：基于样本框架和随机单位如人、家庭等，使用软件随机地分成控制组和实验组。
   2. 分层随机：事先创建组别/层次，将类似的单位放在一起，在每一层里，样本简单随机化。分层可以减少方差。
   3. 聚类随机：和分层相反，创建很多个桶，并将所有的桶随机化。聚类会降低功效，因为同一个桶里的样本不会随机化。
3. 当不可能对样本一次性随机化的时候，有以下方法可以尝试：
   1. 逐步设计(phase-in design)：假设一个NGO表示他们不能联系部分对象单纯做对照组(即所有的对象都必须接受实验),则为了实施RCT，可以采用逐步设计的方法，逐步将对象纳入实验组中。但这一设计有很多不足，现在被尽量避免使用。不足有：
      1. 因为是逐步实验的，长期实验效果较难衡量，因为对照组最终也会接受实验。
      2. 对照组通常知道它将很快接受实验，这可能会改变他们的行为。
         ![img](MIT-Data_Analysis_for_Social_Scientists(II).assets/48B95302-DAA2-4F28-B29F-6ED5387AF96E.jpg)
   2. 在泡泡中随机化(randomization in the bubble)：排除掉完全拒绝/接受的群体，在阈值附近进行分层随机化。
      ![img](MIT-Data_Analysis_for_Social_Scientists(II).assets/46CE61FE-50A4-443F-B5AA-00258D59F7B4.jpg)
   3. 鼓励设计(encouragement design)：在IV中非常流行，如之前提及的利用奖学金研究教育的影响。

### 2.2. question driven designs

1. 实际上，我们在进行实验设计时，要根据研究问题选择随机化的方式。
2. 两步RCT(two-step randomized controlled trial)：可用于评估问题的一般化平衡效果(estimating general equilibrium effects)以及溢出效应(spillover)。
   1. 例如：如果要研究就业促进计划是否有利于降低失业率，则可能出现的问题是：将同一个劳动力市场的对象分成对照组和实验组，则实验组可能会产生替代效果/溢出效果(displacement effect/spillover)，即实验组和对照组竞争同样的岗位，而实验者被雇佣了，说明就业促进政策提高了实验组的就业能力，但可能没有降低该市场的失业率。
   2. 因此，对于该情形，应采用两步RCT：先随机分配各个地区的实验比例，再在地区里随机分配实验对象。
      ![img](MIT-Data_Analysis_for_Social_Scientists(II).assets/D089D640-D9E0-47AA-B321-FF5DEED53538.jpg)

## 3. Data Visualization-Tell Others' Story

1. 数据可视化有两方面目的：
   1. 对自己：探索性数据分析，了解数据以便进行进一步研究
   2. 对他人：跟其他人讲述你的数据和研究成果。
      ![img](MIT-Data_Analysis_for_Social_Scientists(II).assets/2AE9A972-00EC-4406-AD4E-6D8709500BB0.jpg)
      ![img](MIT-Data_Analysis_for_Social_Scientists(II).assets/BDE3CE36-2B4C-4402-B384-33FC6329A152.jpg)

### 3.1. Charts

1. 塔夫特画图原则(Tufte's Principles)：塔夫特是这个领域的专家，不过他提出的很多图形在实践中较少人使用，原因可能是技术实现上的问题，不过记住他的原则有利于我们画出更便于交流的图形。
   1. 展示数据(show the data)
   2. 最大化数据墨水比例(maximize data-ink ratio)：即图表上大部分的笔墨应该用于展示数据。
   3. 尽可能减少非数据墨水(erase non data ink as much as possible)
   4. 减少冗余的数据墨水(erase redundant data ink)
   5. 避免图表垃圾(avoid chart junk)：如纹理、一维数据3D化
   6. 尝试增加数据墨水的密度(try to increase the density of data ink)
   7. 图应该是水平的(graphics should tend to be horizontal)：以便于阅读
      ![img](MIT-Data_Analysis_for_Social_Scientists(II).assets/9A66E52D-EF21-424D-BD5F-520C9B7E3BC5.jpg)
2. 画图规则总结：
   1. 条形图是通过柱状展示数据的，因此，轴坐标必须从零开始。
   2. 避免过多的饼图：因为就视觉上而言，我们很难新人通过扇形和角度判断的数据，一般还需要通过标注的百分比确认，很多时候就造成了展示冗余。
   3. 展示数据时注意：多子图/小倍数(small multiple)、整合、透明度、集中
   4. 图表必须能传达足够的信息(self-explanatory)：
      1. 轴标签、单位
      2. 数据意义和重点

### 3.2. Tables

1. 表格也是要遵循一样的原则：
   1. 展示数据(show the data)
   2. 不说谎(don't lie about it)
   3. 集中信息(focus)
2. 除了以上原则外，表格还应该满足以下规则：
   1. 只展示重要的数字而不是全部
   2. 控制有效数字位数小数点后2位/3位)
   3. 删除竖线，减少横线(no vertical lines, very few horizontal lines)
3. 在R中，可以使用"stargazer"包画出标准化的表格，教学网站：https://www.jakeruss.com/cheatsheets/stargazer/

# Module 11

## 1. Estimation vs ML

1. 机器学习和估计的不同源于目的。统计学的估计致力于找到最佳的系数/估计量，并解释系数的意义，来挖掘y和x之间的关系；而机器学习则是希望预测最接近的y，而不关心系数的准确性。两者的优化目的是完全不同的。
   1. 机器学习的x变量有时会多于数据量(high dimensional)，这在估计中会导致矩阵不可逆，是不可行的。
   2. 机器学习会尝试对原变量(raw variables)进行各种转换，创建各种各样有效的x变量以便准确预测y，而不追求系数是否有意义。比如在情感分析中(sentiment analysis)，将数据中包含的所有单词(unigrams)、连续两单词(bigrams)作为x变量，并在每个数据点中标记是否出现，以此构建句子结构，以便更好地预测y。
   3. 机器学习和估计的折中点(trade-off)在于：机器学习并不要求系数必须是无偏的(not conditional on being unbiased)，以便进行最佳预测。
   4. 也就是说：估计致力于探究x和y之间的关系，找到最佳系数；而机器学习就只是将尽可能多的x丢进去并找到最佳的预测值，而不管哪个x变量真正起了作用。预测模型的工作方式无法理解估计值或赋予它们任何意义，因此我们没有理由期望它们对任何单个变量的影响产生可解释的估计。
      ![img](MIT-Data_Analysis_for_Social_Scientists(II).assets/A7C38C06-02A5-4378-B293-75BF4721A6BE.jpg)
2. 估计和预测(estimation vs prediction)相结合的经典例子是：工具变量法IV，模型的第一步是通过工具变量预测感兴趣的x变量，第二步则是估计x变量的系数。
   1. ML for Estimation：机器学习通过帮助我们生成某些变量，系数和数据，这些变量可以帮助我们进行估计。

## 2. Secret Sause of ML

1. 机器学习的秘密调料一是：将估计的样本内误差最小化(in-sample minimizing)转化为样本外误差最小化(out-of-sample minimizing)，避免过拟合问题，减少模型方差。
   1. 估计：我们只需要找到样本的无偏估计量，因此得到的模型通常是过拟合的。
   2. 机器学习：我们要找到能一般化的模型(generalized model)。
      ![img](MIT-Data_Analysis_for_Social_Scientists(II).assets/943E6B2E-60CB-4675-8215-55D8951AC279.jpg)
2. 机器学习的秘密调料二是：它可以根据数据集和算法决定模型的复杂度(complexity)，算法会决定输出的最优模型。而不像估计，需要一开始就进行模型假设(包括有多少个x变量、是否有工具变量等)。
   ![img](MIT-Data_Analysis_for_Social_Scientists(II).assets/0078A1D7-5058-4F2A-885E-81105D407E26.jpg)