# 前言

> * 本文将介绍如何使用Plotly+Cufflinks更简便地制作出更好的图表。
> * 这是Github上一个大神发布的资料，本人学习之后觉得非常实用，所以和大家分享。需要查看代码的童鞋指路Github：https://github.com/WillKoehrsen/Data-Analysis
> * 本文中的所有代码都是使用Jupyter notebook完成的，在使用pip命令安装了plotly和cufflinks之后，可以import使用它们。

# 1、Plotly+Cufflinks是什么？

* Plotly Python包是Plotly公司开发的可视化软件的开源版本，是基于plotly.js构建的，而后者又建立在d3.js上。
* 因为Plotly不能直接接受numpy和pandas的数据结构，所以用一个名为cufflinks的封装器来使用Pandas数据可以减少数据处理工作。
* 这两个库的组合使用起来很简单，大部分时候可以用**一行代码**生成非常棒的图表，
  会比Matplotlib简单多了。
* 导入库：
  * import plotly.graph_objs as go
  * import cufflinks as cf
  * from plotly.offline import iplot,init_notebook_mode
  * init_notebook_mode(connected=True)
  * cf.go_offline(connected=True)

# 2、花样制作各式图表

> 直方图、箱线图、柱状图、散点图、折线图、饼图、散点矩阵、热力图、散点3D图、气泡3D图，全部都可以用简单的一行代码搞定！

## 2.1、直方图

* 直方图是绘制单一变量的首选图，下图是作者WillKoehrsen绘制的变量['Claps']直方分布图：

![img](Plotly+Cufflinks_Visualization.assets/53805A94-F1E9-4974-A33B-122E6AEA44F4.png)
![img](Plotly+Cufflinks_Visualization.assets/1.gif)

* 代码非常简单，就是在data之后加一个iplot后缀，并添加相应的参数。图表是交互式的，把鼠标放在bins可以获得相应数据。

### 2.1.1、分组直方图

* 绘制分组直方图只需要添加参数[barmode='group']即可，非常简便。
  ![img](Plotly+Cufflinks_Visualization.assets/16559B0B-D41E-4156-8E46-AE10EE9298E8.png)
  ![img](Plotly+Cufflinks_Visualization.assets/2.gif)

### 2.1.2、叠加直方图

* 绘制叠加直方图则添加参数[barmode='overlay']。
  ![img](Plotly+Cufflinks_Visualization.assets/72E57EC3-30B1-456C-911A-65A84AAA673B.png)

### 2.1.3、小结

* 代码：df['value'].iplot(kind='hist',bins= ,xTitle= ,yTitle= ,title= )
* 其他参数：linecolor、opacity（透明度）、bargap（间隔）、histnorm、barmode

## 2.2、柱状图

* 对于条形图，需要先应用聚合函数，将x轴变量设为索引，然后再使用iplot绘图。例如作者以['publication']进行分组并计算变量['fans']的数量，再进行图形展示：
  ![img](Plotly+Cufflinks_Visualization.assets/3F64CC60-BA54-4123-95A6-B1AAFE0DDEF9.png)
* 如果绘制多个分类的柱状图，则相应添加多个y轴变量即可，非常简单！
  ![img](Plotly+Cufflinks_Visualization.assets/F6520E5F-F223-4494-9F15-D2815EC552A2.png)

### 2.2.1、双坐标轴

* 如果两个分组变量的范围相差太大，我们又想把它们放在同一个坐标轴上，则可以设立y2轴。
* 设立y2轴只需要添加参数secondary_y。
  ![img](Plotly+Cufflinks_Visualization.assets/B1B7301A-F855-4577-AC85-95E80838C22D.png)

### 2.2.2、小结

* 代码：df.iplot(kind='bar',xTitle= ,yTitle= ,title= )
* 其他参数：secondary_y、secondary_y_title

## 2.3、箱线图

* 箱线图的制作和直方图类似，不过要把kind参数换成[kind='box']。
  ![img](Plotly+Cufflinks_Visualization.assets/9EBF8E20-031E-420B-8592-C5367E27E3BB.png)
  ![img](Plotly+Cufflinks_Visualization.assets/4.gif)

### 2.3.1、分类箱线图

* 如果我们需要制作分类箱线图，则需要先制作一个透视表。
  ![img](Plotly+Cufflinks_Visualization.assets/1A5C4A98-2981-40FF-915E-1A07BB59F42C.png)

### 2.3.2、小结

* 代码：df.ilpot(kind='box',xTitle= ,yTitle= ,title= )、df.pivot(columns= ,values= )
* 其他参数：colorscale、layout

## 2.4、散点图和折线图

* 制作散点图和折线图的话，和前面3个图不同，需要将kind参数更改为mode参数，不然会报错。
* x轴变量默认为索引，但可以通过参数[x=' ']进行更改。
  ![img](Plotly+Cufflinks_Visualization.assets/B2361374-DDD8-4B82-B616-2325B7C21669.png)

### 2.4.1、增加拟合线

* 增加拟合线相关参数：bestfit=True
  ![img](Plotly+Cufflinks_Visualization.assets/5.gif)

### 2.4.2、增加文字注释

* 利用text参数增加文字注释。
* 作者利用字符串格式化和HTML写了一个例子：
  ![img](Plotly+Cufflinks_Visualization.assets/3A3C1E1B-47E5-4A3B-8B0D-EA09974783F7.png)

### 2.4.3、分类散点图

* 制作分类散点图可以通过categories参数添加：
* 此外，也可以通过size参数对散点做进一步的区分，但size参数所带变量必须是数值变量。
  ![img](Plotly+Cufflinks_Visualization.assets/E0A49F93-E524-43F2-BE54-B8AF9D1F6F1C.png)

### 2.4.4、添加参考区域或参考线

* 使用hline和vline参数可以添加线，使用vspan和hspan参数可以添加区域，和Matplotlib语法是类似的~
  ![img](Plotly+Cufflinks_Visualization.assets/504C1C8E-CD55-4B4E-8965-20B21E6FC243.png)

### 2.4.5、小结

* 相关参数：bestfit、text、categories、symbol（散点形状设置）、size（散点大小）、xrange（x轴范围）、yrange（y轴范围）、hline（水平参考线）、vline（垂直参考线）、hspan（水平参考区域）、vspan（垂直参考区域）

## 2.5、散点矩阵和热力图

* 导入画图库：import plotly.figure_factory as ff
* 散点矩阵画图函数：ff.create_scatterplotmatrix()；热力图画图函数：ff.create_annotated_heatmap()
  ![img](Plotly+Cufflinks_Visualization.assets/8D8CAB6B-1543-43D4-B77A-1CDF58EBFBDB.png)

## 2.6、饼图

* 要制作饼图，需要先用聚合函数对变量进行分类，但不能设置分类变量为索引，否则无法画图。
  ![img](Plotly+Cufflinks_Visualization.assets/DCF8330F-CBAA-44D7-8924-EB8340950458.png)

## 2.7、3D图形

* 除了以上图形，plotly也可以画好看的3D图形，比如曲面图、3D散点图等。
  ![img](Plotly+Cufflinks_Visualization.assets/6.gif)

# 3、总结

> * 比起Matplotlib和seaborn，Plotly可以快速地实现交互可视化，并输出令人愉悦的图形，让我们能更深入地探索数据细节。