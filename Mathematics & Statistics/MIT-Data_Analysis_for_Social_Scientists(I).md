# Module 1

> * data is beautiful, insightful and powerful，however, it could be deceitful(eg: Correlation doesn't mean causality.
> * When considering causality, we might miss some hidden variables that really effect and even reverse causality.

* Like picture below, it might be difficult to see a causal story between these two varibles.
  ![img](MIT-Data_Analysis_for_Social_Scientists(I).assets/C9A9E881-1375-46EF-AEA8-67E7D94609C3.png)
* Unless we go to the data with a **structure and a disciplined way**, the data is going to place tricks to us.

## 1. what we need to learn

* 需要学习的知识：
  1. 对收集数据的过程进行建模——概率论
  2. 总结和描述数据——统计学
  3. 变量之间的关系——探索性数据分析、计量经济学和机器学习
  4. 因果关系分析——因果关系、RCT/AB test、回归
  5. 分析结果展示——图表绘制
  6. 实际操作技能——R/数据收集
     ![img](MIT-Data_Analysis_for_Social_Scientists(I).assets/350D91DA-BDC1-40C3-9496-1BAECAB420DD.png)
     ![img](MIT-Data_Analysis_for_Social_Scientists(I).assets/933871FD-74C5-48C5-A399-4B1FE581426B.png)

# Module 2

> Fundamentals of Probability, Random Variables, Joint Distributions + Collecting Data

## 1. Fundamentals of Probability

### 1.1. Some definitions

1. 样本空间(sample space) ：一个实验中所有可能结果的集合，简称S。取样类型有：

   1. 放回取样(sampling with replacement)

   2. 不放回取样(sampling without replacement)

2. 事件A(event A)：任何结果的集合，包括单个结果、整个样本空间、空值集合(the null set)。

   1. 如果事件中有结果，则认为事件已发生(the event is said to have occurred)
   2. 事件B包含于事件A($Bsubset A$)，说明B中的每一个结果也都属于A(event B is contained in event A)

3. 集合相关定义：

   1. 交集(intersection)：$Acap B$/$AB, quad P(ABcomplement) = P(A) - P(AB)$

   2. 并集(union)：$Acup B, quad P(Acup B) = P(A) + P(B) - P(AB)$ the law of inclusion-exclusion

   3. 补集(complement)：${A}complement$

   4. 空集(null set)：$varnothing, quad P(varnothing) = 0$

4. 集合相关关系：

   1. 互斥(mutually exclusive/disjoint): A和B没有共同的结果，则称AB互斥。

   2. 互补(exhaustive/complementary): A和B的并集为S($P(S) = 1$)，则称AB互补。

5. 排列和组合：
   1. 排列(permutation)：物体的有序排列(ordered arrangement)。从N个物品中抽出n个物品(不放回)的排列方式有$N!/(N-n)!$种；从N个物品中抽出n个物品(放回)的排列方式有$N^n$种![img](MIT-Data_Analysis_for_Social_Scientists(I).assets/3A7BE82E-2D45-48EE-891B-0DFBD2DFD3A9.png)
      2. 组合(combination)：物体的无序排列(unordered arrangement)。从N个物品中抽出n个物品的组合方式有N!/{n!(N-n)!}种，记为$tbinom{N}{n}$。有放回组合计算：
         ![img](MIT-Data_Analysis_for_Social_Scientists(I).assets/0E531D20-D80B-42D6-82C0-828AEC733671.png)
6. 独立性(independence)：如果*P*(*A**B*)=*P*(*A*)*P*(*B*)，则认为事件A和事件B相互独立。
   1. 如果事件A和事件B相互独立，事件A和B的补集也相互独立。

### 1.2. conditional probability

* 条件概率定义：

  1. 事件A在事件B发生的条件下的概率：$P(A|B) = P(AB)/P(B) quad P(B)>0$
  2. 条件概率根据新信息，重新定义了事件样本空间。

* 条件概率和独立性的关系：

  1. 若事件A和事件B相互独立，说明B的发生并没有告诉我们有关A的信息，则：
     $$
     P(A|B) = P(AB)/P(B) = P(A)P(B)/P(B) = P(A)
     $$

### 1.3. Bayes' Theorem

* 贝叶斯定理：利用先验概率和先验条件概率计算后验条件概率。
  $$
  P(A|B) = \frac {P(B|A)P(A)}{P(B|A)P(A)+P(B|{A}\complement)P({A}\complement)}
  $$
  

![img](MIT-Data_Analysis_for_Social_Scientists(I).assets/1824B929-810A-4B34-BD5C-40E74E44A086.png)
![img](MIT-Data_Analysis_for_Social_Scientists(I).assets/2448374F-3749-4449-A7A5-C9267C6B77DD.png)

## 2. Random Variables, Distributions

### 2.1. Random Variables

> 分析样本空间的数字特征如平均值、总和等基于一个重要的数学构造，也就是随机变量(There is an important and useful mathematical construct we exploit to analyse that numerical characteristic, the random varible)。

* 随机变量是样本空间的一个实值函数(real-valued function)，一般写成
  R。随机变量的概率引出X的分布。
  ![img](MIT-Data_Analysis_for_Social_Scientists(I).assets/44E10789-F9F1-4301-B393-EB805152DDBC.png)
* 随机变量分为：
  * 离散型(discrete)，如超几何分布、二项分布，离散型的值之间是分散的(ps：变量个数不一定是有限的)。
  * 连续型(continous)，如正态分布，其概率函数(PF)也称为概率密度函数(probability density function, PDF)。
  * 随着离散型个数n的增加，可近似看成连续型。

### 2.2. PF and PDF

1. 概率函数(probability function, PF)：指计算离散型随机变量每个值的概率的函数，记为$f_X(x)$。对于任何一个值，满足：
   $$
   f_X(x) = P(X=x)
   $$

PF具有以下属性(properties):

* $0≤f_X(x)≤1$
* $sum_i f_X(x) = 1$
* $P(A) = P(X in {A}) = sum_A f_X(x_i) qquad text{@概率的加法}$

2. 概率密度函数(probability density function, PDF)：计算连续型随机变量的概率的函数，与积分有关，亦记为$f_X(x)$。

PDF的特有属性(properties)：

* 对于连续型随机变量X，存在一个非负函数$f_X(x)$使得任意区间A, X在A($A subseteq R$)中的概率等于这个区域的积分(intergral)：
  $$
  P(X \in A) = \int_A f_X(x)dx
  $$

1. PF和PDF的区别：

* PDF大于等于0，但不一定小于1，在特定区间内，可能大于1，即 $0 <= f_X(x)$，**也就是说，不能将概率密度函数上的一个点认为是概率**。举例：均匀分布X~U(0.0.5)。
  ![img](MIT-Data_Analysis_for_Social_Scientists(I).assets/EBE36715-CACF-4AA7-8F13-BC1CA9F251C6.png)
* F的各个概率值相加等于1；而PDF的积分等于1，即$int f_X(x) = 1$
* 对于单个值x，离散型变量有特定的概率值$P(X=x)$；而连续型变量单个值的**概率**则为0(PS：概率 ≠ PDF的值)，计算PDF必须以区间为单位。

### 2.3. CDF

> 在实际工作中，使用累积分布函数(cumulative distribution function)了解变量分布会比PF和PDF方便得多。

1. 1. 累积分布函数的定义：对于随机变量X，其CDF记为$F_X$，满足：

   $$ F_X(x) = P(X<= x)$$

2. CDF的属性(properties)：

   * $0 <= F_X(x) <=1$

   * $F_X(x)$是不减函数(non-decreasing)

   * $lim_{nrightarrow+infty} F_X(x) = 1$

   * $lim_{nrightarrow-infty} F_X(x) = 0$

   * 对于连续型变量，CDF是连续的；对于离散型变量，CDF会在变量处跳跃。

3. CDF和PDF的转换：积分和微分![img](MIT-Data_Analysis_for_Social_Scientists(I).assets/18D05C23-8A25-4546-A616-8885D410FFAD.png)

## 3. Gathering and Collecting Data

### 3.1. Data Sources

* 我们可以从3个方面得到数据：
  * 数据图书馆(existing data libraries)
  * 收集自己的数据(collecting your own data)
  * 从网上抓取数据(extracting data from the internet)

### 3.2. Existing Data Libraries

![img](MIT-Data_Analysis_for_Social_Scientists(I).assets/FD4EC86C-A22B-491A-8B09-19DA7ED8FEDD.png)

### 3.3. Web Scraping

* 网络抓取数据(web scraping)
  * API：通过应用程序接口获得
  * 使用Python：BeautifulSoup包
  * 使用R：rvest包、谷歌插件SelectorGadget
* 使用R进行抓取的代码展示：
  ![img](MIT-Data_Analysis_for_Social_Scientists(I).assets/82845EC1-59DD-416C-B0F9-127461E9DDD2.png)

```
# 抓取edx上的课程名称
library(rvest)

 edxsubjects <- read_html("https://www.edx.org/subjects")
subjectshtml<-html_nodes(edxsubjects, ".align-items-center")
subjecttext<-html_text(subjectshtml)

print(subjecttext)

# 抓取旧书网的书籍和价格并进行数据清理
library(rvest)
library(tidyr)
library(dplyr)

larecherche <- read_html("https://www.abebooks.com/servlet/SearchResults?pt=book&sortby=17&tn=a+la+recherche+du+temps+perdu&an=proust&cm_sp=pan-_-srp-_-ptbook")

price <- larecherche %>%
  html_nodes(".srp-item-price") %>%
  html_text() %>%
  readr::parse_number() #提取数字

title <- larecherche %>%
  html_nodes("h2 span") %>%
  html_text() %>%
  readr::parse_character() #提取字母

combined <- data_frame(title, `data and time`=Sys.time(), price)
print(combined)
```

### 3.4. Collecting Your Own Data

![img](MIT-Data_Analysis_for_Social_Scientists(I).assets/B6AD0C3F-894A-4E04-9108-DAD01A10D082.png)

* 收集自己数据的时候，需要遵守相关的法律法规，并保护好人类受试者(human subject)。
  ![img](MIT-Data_Analysis_for_Social_Scientists(I).assets/22C57C84-9E3F-4648-9261-5298CB0DBAB8.png)
* 贝尔蒙特原则(Belmont Principles)：
  1. 对人的尊重(respect for persons)：获得知情同意并保护受试者
  2. 善良(beneficence)：确保试验的危害最小
  3. 公平(justice)：确保选择主体的公平性

# Module 3

## 1. Visualizing & Describing Data

> 探索性数据分析(exploratory data analysis)指根据研究目的，对数据进行总结和描述。

### 1.1. Distribution Graphs

1. 直方图(histogram)：直方图通过对数据进行分桶(bins)，对变量的总体分布(underlying distribution)进行粗略估计。
   1. 画图设置：桶的数量(number of bins)、y轴的统计(选择统计数量或百分比：count/proportion)
   2. 图形缺点：因为直方图是长方形堆叠，当数据过少时，它是崎岖而不连贯的(bumpy)，不利于直观展示数据分布。

```
# 1.画一个简单的直方图
ggplot(data, aes(height))+
    geom_histogram(fill = 'blue', color = 'darkblue'， binwidth = 5)+
    xlab('Height')
    ylab('Counts')

# 2.画一个密度直方图
ggplot(data, aes(height))+
    geom_histogram(aes(height, ..density..), fill = 'blue', color = 'darkblue'， binwidth = 5)

# 3.画一个曲线直方图(取桶中点进行连线)
ggplot(data, aes(height))+
    geom_freqpoly(aes(height, ..density..), fill = 'blue', color = 'darkblue'， binwidth = 5)
```

1. 核密度估计图：核密度估计图被认为是更光滑的直方图，解决了直方图的缺点，是估计随机变量的概率密度函数的非参数方法(non-parametric way to estimate the PDF of a random variable)
   1. 画图设置：不同与直方图的桶划分，核密度估计图应用核函数(对称并集中在1点即对应的数据点)可得到每个数据点的核函数图形，并设置带宽(bandwidth)，计算对应带宽的全部核函数高度的总和，可以得到最终图形。
      ![img](MIT-Data_Analysis_for_Social_Scientists(I).assets/A8089EDA-4348-49DB-9C2E-A0E2DB76A612.png)
   2. 核函数：核函数有很多选择(Epanechnikov曲线、高斯曲线等)，选择哪个都可以，不会有太大影响。但它一般是钟形的、积分为1。数据点的权重最高，接着沿两边不断降低。
      ![img](MIT-Data_Analysis_for_Social_Scientists(I).assets/78DE3E30-E0BF-4710-B7FD-FAEC79731E42.png)
   3. 带宽：太小则曲线太弯曲，太大则曲线更顺滑，但可能忽略了一些分布特征，最优带宽使平方误差最小化。
      ![img](MIT-Data_Analysis_for_Social_Scientists(I).assets/533973A1-CEC1-4EE8-9F1F-E5BA1410889B.png)

```
# 画一个核密度估计图+密度直方图
ggplot(data, aes(height))+
    geom_histogram(data, aes(height, ..density..), fill = 'blue', color = 'darkred')+
    geom_density(data, kernel = 'gaussian', aes(height), bw = 1) # 不设置bw的话，R会提供最优的bw
```

![img](MIT-Data_Analysis_for_Social_Scientists(I).assets/042EB53E-B4D6-4E07-8E04-0B2838D82F6D.png)

1. 估计累积概率密度图(ECDF)：当比较两个变量分布时，不仅可以使用核密度估计图，也可以使用eCDF图。相比之下，eCDF会更方便比较两者之间的分布差异。
   1. ecdf：可以直观比较两条曲线的高度(cdf)、高度差(pdf)和上升趋势，从而了解变量间的分布差异。比如：印度和美国低于1.6米的女性，哪个国家的比例高？——我们只需要比较曲线高度。
   2. kdf：可以直观比较两者的分布对称性、分散程度。
      ![img](MIT-Data_Analysis_for_Social_Scientists(I).assets/2BD39C42-CABC-47E1-AD54-F1287E71FD7A.png)

```
# 绘制累积密度图
ggplot(data, aes(height)+
    stat_ecdf(data, aes(height), color = 'darkblue')
```

## 2. Special distributions

### 2.1. Joint Distribution

1. 联合分布是研究变量间关系的重要工具，如降雨量和庄稼、快速结账队伍和常规结账队伍长度等。常见的联合分布：

   * 二元分布(bivariate distributions)：只有两个随机变量。
   * 多元分布(multivariate distributions)：多于两个随机变量。

2. Joint PDF和Joint PF：

   1. 联合概率密度函数的定义(Joint PDF)：联合概率密度函数的定义(Joint PDF)：如果X和Y是定义在同一样本空间S的连续随机变量，则X和Y的联合概率密度函数(Joint probability density function)记为$f_{XY}(x, y)$，在任意区间A内需要计算XY平面的积分，即对X和Y进行二重积分：

      $P((x,y) \in A) = \int \int_A f_{XY}(x, y)dxdy$![img](MIT-Data_Analysis_for_Social_Scientists(I).assets/A26D9B64-DA8D-4B31-91A6-04EB14894BC7.png)
      ![img](MIT-Data_Analysis_for_Social_Scientists(I).assets/6C2EDA4D-937D-4FF3-AFDE-B572B46D72A3.png)

   2. 联合概率函数的定义(Joint PF):对于离散型变量X和Y，联合概率函数(joint PF)满足：

      $f_{XY}(x, y) = P(X=x \,and \, Y=y)$![img](MIT-Data_Analysis_for_Social_Scientists(I).assets/ED503815-B482-46A1-9075-F51DF1F5D7D6.png)

### 2.2. Marginal Distribution

1. 假设我们得到了X和Y的联合分布，但我们实际上只关心X/Y的分布，即X/Y在该联合分布的边际分布(marginal distribution/individual distribution)。

2. 边际分布的PF和PDF：

   1. 对于离散型变量：对X特定的值，将所有的y值PF相加即可得到该值的边际分布。
      $f_X(x) = \sum_y f_{XY}(x,y)$
      ![img](MIT-Data_Analysis_for_Social_Scientists(I).assets/784EAD11-F374-4214-BCFA-36CCA4A9ED26.png)

   2. 对于连续型变量：对X特定的区间，对y进行积分。

      $f_X(x) = \int_y f_{XY}(x,y)dy$
      ![img](MIT-Data_Analysis_for_Social_Scientists(I).assets/724117E0-EA28-4C63-B0E8-90123247066C.png)

3. 如果我们知道所有变量的边际分布，以及变量间的关系，才可以得到变量的联合分布。也就是说，**要构建联合分布，需要知道边际分布和变量间的关系，两者缺一不可**。

### 2.3. Independence of RVs

1. 随机变量之间的独立性是指：变量A和变量B的概率乘积是变量AB的发生概率，即：

   $P(A) \cdot P(B) = P(AB)$

2. 证明独立性的方法有以下三种：

   1. CDF：$F_{XY}(x,y) = F_X(x) \cdot F_Y(y)$
   2. PDF：$f_{XY}(x,y) = f_X(x) \cdot f_Y(y)$
   3. 非负函数：$f_{XY}(x,y) = g(x) \cdot h(y) \quad \text{注：g和h分别是X和Y的非负函数}$![img](MIT-Data_Analysis_for_Social_Scientists(I).assets/EC10D2EA-B62C-41A0-A9A5-32974779B590.png)
      ![img](MIT-Data_Analysis_for_Social_Scientists(I).assets/51103BF6-9B75-461F-980F-45BCCAE18B7F.png)

### 2.4. Conditional Distribution

1. 和条件概率定义相似，即在给定变量间关系的条件下( a conditional distribution was like a way of updating the distribution
   of a random variable given some relevant information)，可以得到变量的条件分布。
   1. 离散型：$P(Y=y|X=x) = \frac{P(Y=y, X=x)}{P(X=x)}$
   2. 连续型：$f_{Y|X}(y|x) = \frac{f_{XY}(xy)}{f_X(x)}$
      ![img](MIT-Data_Analysis_for_Social_Scientists(I).assets/67683806-01CB-406B-8C35-F72D0C90621E.png)
2. 当变量X和Y相互独立(independent)，Y相对于X的条件分布就是Y的边际分布。
   ![img](MIT-Data_Analysis_for_Social_Scientists(I).assets/1A10FA49-01F7-4095-909F-F090B4853702.png)
   ![img](MIT-Data_Analysis_for_Social_Scientists(I).assets/937F3273-5C31-4E69-998D-16F70DFF37A7.png)

# Module 4

## 1. Functions of RVs

1. 当我们谈论统计学，我们其实就是在讨论随机变量的函数。因此，我们需要了解随机变量及其函数的相关性质。
2. 取决于随机变量的类型、函数是否可逆等多种因素，我们有很多种方法可以求得随机变量函数的分布，在这里我们介绍最常用的一种。

### 1.1. Method to get distribution of RVs' Function

1. 假设X是分布为$f_X(x)$的随机变量，Y = h(X)，则Y的分布为：
   $$
   F_Y(y) = \int f_X(x)dx = P(Y <= y) = P(h(x)<= y)\\
   \text{注：根据h(x)<=y可得到X的取值范围}
   $$

* 若Y是连续变量，则：
  $$
  f_Y(y) = dF_Y(y)/dy
  $$

![img](MIT-Data_Analysis_for_Social_Scientists(I).assets/image-20200730153554104.png)

![img](MIT-Data_Analysis_for_Social_Scientists(I).assets/998E8DCE-E3AB-4FDD-AD58-39CCB6D581EA.png)

* 若Y和X都是离散变量，则：
  $$
  f_Y(Y=y) = P(h(x)=y)= P(X=g(y))
  $$

![img](MIT-Data_Analysis_for_Social_Scientists(I).assets/3FB05278-0BBE-4468-97E6-0FD4285D50EE.png)
![img](MIT-Data_Analysis_for_Social_Scientists(I).assets/6B38A1EC-C2F9-4F6F-A88E-9DF757CAD716.png)

### 1.2. Important Examples of RVs‘ Function

1. 线性变换(linear transformation)：假设X是分布为$f_X(x)$的随机变量，Y = aX+b ，则Y的分布为：
   $$
   f_Y(y) = \frac{1}{|a|}f_X(\frac{y-b}{a})
   $$

![img](MIT-Data_Analysis_for_Social_Scientists(I).assets/image-20200730160234963.png)

2. 概率积分变换(probability integral transformation)：假设X是PDF为$f_X(x)$、CDF为$F_X(x)$的连续型随机变量，Y = $F_X(x)$ ，则Y的取值在[0,1]之间，且$F_X$是可逆的(再次强调：X是连续型随机变量)，Y的分布为：
   $$
   F_Y(y) = P(F_X(x) <= y) = P(X<= F_X^{-1}(y))= F_X(F_X^{-1}(Y))= y \quad \{0<=y<=1\}
   $$

* 从中可以看出来，y满足均匀分布[0,1]。亦就是说，任何一个连续型随机变量的CDF变换满足U[0,1]的均匀分布。概率积分变换的应用有：

  1. 电脑模拟从特定分布中随机抓取变量值(random draws from the specific distribution)，比如**若要创建一个符合指数分布/beta分布的随机变量(很多编程语言没有现成的函数直接生成)，那么可以计算这些分布CDF的反函数，并使用该函数将从U[0,1]中随机抓取的值转换成指数/beta分布，从而得到伪随机数(pseudorandom number)**。即：
     $$
     X := F_Y^{-1}(U)
     $$

  2. 反函数定义：设函数y=f(x)(x∈A)的值域是C，若找得到一个函数g(y)在每一处g(y)都等于x，这样的函数x= g(y)(y∈C)叫做函数y=f(x)(x∈A)的**反函数**，记作$f_1(x)$ 。最具有代表性的反函数就是对数函数与指数函数。
     ![img](MIT-Data_Analysis_for_Social_Scientists(I).assets/image-20200730234804562.png)

![img](MIT-Data_Analysis_for_Social_Scientists(I).assets/0096A5F4-F92E-414E-A508-7C7D41324360.png)

```
# 1. 生成标准均匀分布
U <- runif(1000)
# 2. 利用累计函数反函数得到指定分布的随机变量
N <- qnorm(U) #生成标准正态分布
```

3. 概率背景下的卷积(convolution in context of probability)：我们会有很多场景对卷积感兴趣，比如两种投资组合的总收益、n个试验的总成功次数等。卷积可以用两种方式概括：
   1. N个独立随机变量的和(sum of n independent RVs)
   2. 独立随机变量的线性函数(linear function of independent RVs)

4. 假设连续型随机变量X和Y相互独立，Z = X+Y，则Z的分布为：

* XY Joint PDF：

  $f_{XY}(x,y) = f_X(x) f_Y(y)$

* Z的CDF：

  $F_Z(z) = P(x+y < z) = \int_{-\infty}^{+\infty} \int_{-\infty}^{z-y} f_X(x) f_Y(y)dxdy$

* Z的PDF：

  $f_Z(z) = \int_{-\infty}^{+\infty}f_X(z-y) f_Y(y)dy \quad {-\infty <Z < +\infty}$

5. n阶统计量(nth order statistic)：假设X1...Xn是满足同一PDF函数$f_X$的连续性独立且相同分布的随机变量，即一组iiid random variables，也称为随机样本(random sample)。Y = max{X1,X2,...,Xn}，则称Y/Xn为n阶统计量。

   1. 一阶统计量X1也称为样本最小值(minimum)

   2. n阶统计量Xn也称为样本最大值(maximum)

   3. 通过n阶统计量可以计算中位数/均值/四分位数等信息

   4. n阶统计量和一阶统计量的分布分别为：

   $$
   f_n(y) = n(F_X(y))^{n-1}f_X(y)\\
   $f_1(y) = n(1-F_X(y))^{n-1}f_X(y)$
   $$

![img](MIT-Data_Analysis_for_Social_Scientists(I).assets/image-20200730195940905.png)
![img](MIT-Data_Analysis_for_Social_Scientists(I).assets/image-20200730204429836.png)
![img](MIT-Data_Analysis_for_Social_Scientists(I).assets/B3C3365D-2687-4EBF-B8DD-7FB1E2AA82EB.png)

## 2. Moments of a Distribution

1. 分布的矩(moments)帮助我们总结一个分布最重要的一些特征，比如它的均值/中位数/众数/是否对称等。

   1. 正态分布的矩：
      ![img](MIT-Data_Analysis_for_Social_Scientists(I).assets/EAA07C44-7E40-4790-B96D-9586B1477939.png)

2. 通过PDF如何找出均值/中位数/众数呢？

   1. 众数：图形的峰值对应的X，即最大的PDF对应X。

   2. 中位数：PDF$的积分CDF等于0.5对应的X

   3. 均值：**定义域内PDF和X乘积的积分**，即：
      $$
      E(X) = \mu = \int X f_X(x)dx \quad \text{连续型}\\
      E(X) = \mu = \sum X \cdot f_X(x) \quad \text{离散型}\\
      \text{均值叫法有：mean/expectation/expected value}
      $$

### 2.1. Expectation

1. 期望的公式：$E(X) = \mu = \int X f_X(x)dx$
2. 假设Y=r(X)，则：$E(Y) = \int r(X) f_X(x)dx$
3. 期望的属性有：
   1. 一个常数的均值仍是该常数，$E(a) = a$
   2. Y = aX+b，则$E(Y) = aE(X)+b$
   3. Y = X1+X2...+Xn，则$E(Y)= E(X_1)+E(X_2)+...+E(X_n)$
   4. 如果X和Y相互独立，则$E(XY) = E(X)E(Y)$

### 2.2. Variance

1. 方差Var有时也写成$\sigma$，公式：$Var(X) = E((X- \mu)^2) = E(X^2) - E(X)^2 ≥0$

2. 假设Y=r(X)，则：

   $Var(Y) = E(Y^2) - E(Y)^2 = E(r(X)^2) - E(r(X))^2 = \int {r(X)}^2 f_X(x)dx - {(\int r(X) f_X(x)dx)}^2$

3. 方差的属性是：
   1. 常数的方差为0，$Var(a)=0$
   2. Y= aX+b，则：$Var(Y) = a^2 Var(X)$
   3. Y = X1+X2...+Xn，且X之间相互独立，则$Var(Y)= Var(X_1)+Var(X_2)+...+Var(X_n)$

![img](MIT-Data_Analysis_for_Social_Scientists(I).assets/EAE0F40E-B298-46C9-AD91-F532EDE3FD00.png)

1. 比起方差，标准差和随机变量同单位，更便于衡量分布的分散性：
   $$
   SD(X) = \sigma = \sqrt{Var(X)}
   $$

### 2.3. Conditional Expectation

1. 条件均值指的是条件分布的均值，公式为：$E(Y|X) = \int_y F_{Y|X} (y|x)dy$
2. 条件均值相关定理：
   1. 期望迭代定理(law of iterated expectations)：$E(E(Y|X)) = E(Y)$
   2. 总方差定理(law of total variance)：$Var(E(Y|X)) + E(Var(Y|X)) = Var(Y)$
      ![img](MIT-Data_Analysis_for_Social_Scientists(I).assets/AF5B11B4-39BE-4351-B54D-204D3375EAFD.png)
      ![img](MIT-Data_Analysis_for_Social_Scientists(I).assets/FC17208E-4636-475D-9851-20CCD3C25410.png)

### 2.4. Covariance and Correlation

1. 协方差描述了两个变量之间的相关性，当两个变量相互独立时，协方差为0；两个变量相关性越高，协方差绝对值越大。
   $$
   Cov(X,Y) = \sigma_{XY} = E((X-\mu_X)(Y-\mu_Y))
   $$
   ![img](MIT-Data_Analysis_for_Social_Scientists(I).assets/CDEBD4CD-6C8A-44AC-9940-9ACA7ADFF8EA.png)

2. 相关系数(correlation)：协方差的取值范围比较广，受变量单位干扰，不方便进行比较度量，因而使用相关系数度量，取值[-1,1]。
   $$
   \rho (X,Y) = \frac{Cov(X,Y)}{\sqrt{Var(X)} \sqrt{Var(Y)}}
   $$

3. 协方差的相关属性：

   1. $Cov(X,X) = Var(X)$
   2. $Cov(X,Y) = Cov(Y,X)$
   3. $Cov(X,Y) = E(XY) - E(X)E(Y)$
   4. $Cov(aX+b,cY+d) = ac Cov(X,Y)$
   5. $Var(X+Y) = Var(X) +Var(Y) + 2Cov(X,Y)$![img](MIT-Data_Analysis_for_Social_Scientists(I).assets/07A8DECB-51E4-49AF-9A53-2860DE64992B.png)

### 2.5. Two Inequality

1. 马尔可夫不等式(Markov Inequality)：假设X是一个非负的随机变量，则对于任意的t>0, 满足：
   $$
   P(X≥t) ≤ \frac{E(X)}{t}
   $$

![img](MIT-Data_Analysis_for_Social_Scientists(I).assets/CE7757C8-30E4-4741-BBC8-3C5C4FF793E4.png)

1. 切比雪夫不等式(Chebyshev Inequality)：假设随机变量X的方差为*V**a**r*(*X*)，则对于任意的t>0，满足：
   $$
   P(|X-E(X)| ≥ t) ≤ \frac{Var(X)}{t^2}
   $$

![img](MIT-Data_Analysis_for_Social_Scientists(I).assets/D4B43400-FE6F-417A-A7A5-06735FDBACDF.png)

