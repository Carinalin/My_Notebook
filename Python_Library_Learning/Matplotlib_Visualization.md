> * matplotlib：用于数据可视化的基础模块。
> * scikit-learn：机器学习专用库，提供了完善的机器学习工具箱。
> * seaborn：用于绘制更加精致的图形。
> * scipy：数值分析，例如线性代数，计分和插值等。
> * statsmodels：常用于统计建模分析。

***导入绘图数据库：import matplotlib.pyplot as plt***

# 1、matplotlib绘图基础

![img](Matplotlib_Visualization.assets/5DA544B2-4B14-48A5-BA12-A3A4322BA307.png)

1. 保证Jupyter正常显示图形：

* %matplotlib inline

1. 画个简单的图形：

* plt.plot（x,y,lw,'c,marker,ls',markersize,markeredgecolor,markerfacecolor,label）

![img](Matplotlib_Visualization.assets/89E828AE-260D-4C26-A39F-E6830898C3BE.png)

3. 解决中文乱码问题：

* plt.rcParams['font.sans=serif']=['SimHei']>用来正常显示中文标签
* plt.rcParams['axes.unicode_minus']=False>用来正常显示负号

4. 画带有图例的图形

* plt.legend(loc='best')
  ![image-20210407223940506](Matplotlib_Visualization.assets/image-20210407223940506.png)

5. 画布/标题/xy轴设置/储存图片
   ![img](Matplotlib_Visualization.assets/DC0487AE-589F-4DB2-BCB4-57A7F16F0F1D.png)

# 2、简单图形绘制

## 2.1、饼图：plt.pie

![img](Matplotlib_Visualization.assets/28EBECD8-D101-4FF6-A14A-D16B276853B8.png)
![img](Matplotlib_Visualization.assets/7BB9533C-AC19-4850-A768-1FAFE2D3D135.png)

## 2.2、条形图：plt.bar

![img](Matplotlib_Visualization.assets/AF64E39B-7F8F-4D51-AB1A-1D0141EBE609.png)
![img](Matplotlib_Visualization.assets/0ADBDA2D-46EF-4B82-A5CF-552015FC4651.png)

## 2.3、直方图：plt.hist

![img](Matplotlib_Visualization.assets/A6AFF254-9AA7-48DA-980D-504F939C6628.png)

![img](Matplotlib_Visualization.assets/D93B7DBC-47DA-4EF9-8F73-B96F208A71A7.png)

## 2.4、散点图：plt.scatter

![img](Matplotlib_Visualization.assets/85343ED5-BA27-4348-9A08-06D0E600A5A0.png)

![img](Matplotlib_Visualization.assets/39651137-03CE-45CA-AE5A-1B6D9B734928.png)

# 3、图形基本设置

* 图例设置：plt.legend
* 获取画布并修改：plt.gcf、plt.figure
* 设置网格线：plt.grid
* 设置x、y轴的参考线：plt.axvline、plt.axhine
* 设置x、y轴的参考区域：plt.axvspan、plt.axhspan
  ![img](Matplotlib_Visualization.assets/9CFCA43E-5F3E-4040-87C3-13E4E7F16332.png)

# 4、统计图形实战

## 4.1、柱状图

![img](Matplotlib_Visualization.assets/12292026-B27D-497A-85E0-158E2DECF0EF.png)

* 结合pd.pivot_table和pd.crosstab函数制作堆叠的柱状图。
* 采用bottom参数进行堆叠。

![img](Matplotlib_Visualization.assets/18328FF1-D72D-48E7-BCE7-3BB92052A457.png)

![img](Matplotlib_Visualization.assets/ED02A059-FC09-4889-B1C6-8AE011F78759.png)

## 4.2、直方图

![img](Matplotlib_Visualization.assets/B8526B72-3068-472D-A44C-9A8AB4B6C301.png)

* 直方图的绘制和bins的划分息息相关，bins不同，绘制出的直方图会有很大区别。
* 直方图内同一个bin中的样本具有相等的概率密度，增加bin的数量有利于更精确地展现变量的分布。
* 当bin增加到样本最大值时，样本中未出现的值概率为0，导致概率密度函数不连续，因此，可以借助核密度图进行判断（Kernel Density Estimation(KDE)）。
  ![img](Matplotlib_Visualization.assets/BCF17ED2-8139-433C-B203-CE1501DC6B70.png)

## 4.3、箱线图

![img](Matplotlib_Visualization.assets/AD7C6FF3-E05D-4C4A-8F7E-F46BD52DC273.png)
![img](Matplotlib_Visualization.assets/27F445C6-E13B-4D72-B653-37DE6B49229A.png)

## 4.4、散点图

![img](Matplotlib_Visualization.assets/10DE4FC0-B252-468D-859B-60500C1FB930.png)

* 采用for循环进行多类型散点图制作
  ![img](Matplotlib_Visualization.assets/B3950526-AFCA-4D8D-AF9D-08DC9B7D52FD.png)

## 4.5、折线图

![img](Matplotlib_Visualization.assets/EF860414-2B5E-4956-8911-7A7D0DE714B1.png)

# 5、完善统计图形

![img](Matplotlib_Visualization.assets/EAF7983F-613B-4D99-9DAC-86E5738E42D5.png)

## 5.1、图例使用

![img](Matplotlib_Visualization.assets/E34C3447-74DF-4FFB-977F-109D75911618.png)
![img](Matplotlib_Visualization.assets/7961621E-6463-457F-9B81-A7E97189B023.png)

## 5.2、画布移动

![img](Matplotlib_Visualization.assets/14010E57-626B-45B2-8A68-77D95827A9CE.png)

## 5.3、标题设置

![img](Matplotlib_Visualization.assets/B2BE7AA8-0E65-40D7-A0B8-92A11A543989.png)

## 5.4、图形添加内容

![img](Matplotlib_Visualization.assets/CD9A522E-212E-4715-92FD-233A935C7FC4.png)
![img](Matplotlib_Visualization.assets/52AC0011-F3B1-4EE6-9126-088E0BE63607.png)

# 6、图形样式高级操作

## 6.1、绘制双坐标轴

![img](Matplotlib_Visualization.assets/9DB8DF5A-9331-406A-9A59-D00C123DE82B.png)

## 6.2、绘制多个图形

* 采用subplot函数，简单易懂。
  ![img](Matplotlib_Visualization.assets/ED586C02-9A07-4C95-B63A-7138A19174C1.png)
* 采用subplot2grid函数，和subplot类似，只是区域规划机制不同。
  ![img](Matplotlib_Visualization.assets/AF36A845-5455-4A50-8DCF-E0B7AB2F18F8.png)
  ![img](Matplotlib_Visualization.assets/A486677C-33BC-4F78-B291-FDAA5C64098A.png)