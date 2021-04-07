> * seaborn是在matplotlib基础上发展的，使作图更加容易，更容易绘制精致的图形。

# 1、Seaborn基础

![img](Seaborn_Visualization.assets/6FE09374-3BCE-473C-ADE3-8F5C01CDFF86.png)
**导入数据库：import seaborn as sns**

## 1.1、使用方式1：plt.style.use('seaborn')

* seaborn会自动帮忙调整图形，使图形更加美观，比如美化图形背景等。
* plt.style.use('seaborn') 会重置plt函数，因此，防止中文乱码的两行代码需要放在该函数之后。

![img](Seaborn_Visualization.assets/ACDB49F1-B5BE-403D-BEC0-2236ECE9E220.png)

## 1.2、使用方法2：sns.set()

* sns.set的常用参数有style（绘图风格）、context（绘图元素的规模）、Palette（调色板）、

![img](Seaborn_Visualization.assets/D388D90A-D8AE-4E35-B93A-63064CC2B24E.png)

![img](Seaborn_Visualization.assets/6B464F9D-0BB8-44F8-8862-3F68E4905E25.png)

## 1.3、使用方法3：sns.barplot()

* seaborn会自动配置x轴和y轴名称，无需设置plt.xlabel和plt.ylabel。
  ![img](Seaborn_Visualization.assets/B730CCFF-8EDA-48B0-A6B1-940C3364B323.png)
  ![img](Seaborn_Visualization.assets/504A54E3-8449-4BDA-87EE-5F537084EC25.png)
* 如何进行数据标注？
  for x,y in enumerate(GDP.GDP):
  plt.text(x,y+0.1,'%s(万亿)'%round(y,1),ha='center',fontsize=12)

# 2、绘制常用统计图形

## 2.1、柱状图：sns.barplot

* 常用调色板参数有：muted，RdBu，Set1，Blues，husl等，一般不需要设置。

![img](Seaborn_Visualization.assets/83D49B64-2318-4E77-A5BA-77760816086C.png)
![img](Seaborn_Visualization.assets/FACAB720-E98E-4AF8-9F93-0A75501ED678.png)

## 2.2、散点图：sns.scatterplot

![img](Seaborn_Visualization.assets/294018B7-CB7B-4044-8E63-0A45DCC7F758.png)
![img](Seaborn_Visualization.assets/4A83B01D-2B62-423F-8EF4-5E224D3B09EF.png)

## 2.3、箱线图：sns.boxplot

![img](Seaborn_Visualization.assets/6676FB05-CF5A-488E-88C0-7D20998FC8CF.png)
![img](Seaborn_Visualization.assets/3A7BC94F-F6E3-4E6B-82BB-CA993891AA3A.png)

* seaborn绘制的箱线图默认会绘制异常点、平均值和中位数线，因此这些参数可以不用设置。
  ![img](Seaborn_Visualization.assets/9591F1AC-5697-406C-A288-3A8F7D48E764.png)
* 若需绘制分类箱线图，则添加x轴。
  ![img](Seaborn_Visualization.assets/F0A9E903-CE7C-4DB1-95C5-32CB2DFA8671.png)

## 2.4、直方图：sns.distplot

* sns.distplot既可以绘制直方图，也可以绘制kds（核密度图），fit（概率密度图）。
* 要绘制正态分布图需要先导入norm函数包：from scipy.stats import norm

![img](Seaborn_Visualization.assets/A88E7667-1F85-42E5-B8F5-C007683A55D1.png)
![img](Seaborn_Visualization.assets/41E4FCBA-6FA6-45D3-A37F-D69D53E1474F.png)

## 2.5、折线图：sns.lineplot

* plt.xticks(ticks=, labels= )中的ticks是位置列表，用于x轴上的显示限制，如range(0,20,3)；label是指定显示在x轴上的标签，两者均需填写。
  ![img](Seaborn_Visualization.assets/629AE6F1-B9C0-4E14-8809-F79663507123.png)

## 2.6、回归图：sns.lmplot

![img](Seaborn_Visualization.assets/E6E6A7D1-AF69-4C4F-9FED-9761B3BE56A6.png)

# 3、其他参数

## 3.1、计数图：sns.countplot

* 使用matplotlib函数还需要先进行value_counts才能绘制，而sns.countplot可直接绘制。

![img](Seaborn_Visualization.assets/1A89BD14-711C-436D-AC31-4A47831764DB.png)
![img](Seaborn_Visualization.assets/15044A5C-1BAE-4A25-A055-9369ED3AAD3F.png)

## 3.2、风格调整：sns.set

* 设置style：darkgrid , whitegrid , dark , white 和 ticks，默认drakgrid。
* 设置palette：deep, muted, pastel, bright, dark, colorblind
* 设置font_scale：整体放大或缩小字体
  ![img](Seaborn_Visualization.assets/B5DC5C74-096A-48E4-8867-C3D7932C6BB3.png)

