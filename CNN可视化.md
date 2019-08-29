**CNN可视化**

[TOC]



    Markdown Revision 1;
    Date: 2019/07/23
    Editor: Tan
    Contact: scutjy2015@163.com



参考文献：胡秀. 基于卷积神经网络的图像特征可视化研究[D]. 

​	对 CNN 模型的可解释性问题，也称之为深度可视化问题[35]。目前深度可视化方法主要分为两大类，一类通过前向计算直接可视化深度卷积网络每一层的卷积核以及提取的特征图，然后观察其数值变化。一个训练成功的 CNN 网络，其特征图的值会伴随网络深度的加深而越来越稀疏。

​	另一类可视化方法则通过反向计算，将低维度的特征图反向传播至原图像像素空间，观察特征图被原图的哪一部分激活，从而理解特征图从原图像中学习了何种特征。经典方法有反卷积(Deconvolution)[36]和导向反向传播(Guided-backpropagation)。这两种方法能够在一定程度上“看到”CNN 模型中较深的卷积层所学习到的特征。从本质上说，反卷积和导向反向传播的基础都是反向传播，即对输入进行求导。二者唯一的区别在于反向传播过程中经过 ReLU 层时对梯度的处理策略不同。虽然借助反卷积和导向反向传播方法，能够了解 CNN 模型神秘的内部，但这些方法同时把所有能提取的特征都展示出来了，而对类别并不敏感，因此还不能解释 CNN 分类的结果。



​	为了更好地理解 CNN，近几年有大量研究工作都在对 CNN 所学到的内部特征进行可视化分析。最初的可视化工作由多伦多大学的 Krizhevshy 等人在 2012 年提出的AlexNet 一文中出现[20]。在这篇开创深度学习新纪元的论文中，Krizhevshy 直接可视化了第一个卷积层的卷积核，如图 1-4 所示[20]。最早使用图片块来可视化卷积核是在 RCNN论文中出现[37]。Girshick 的工作显示了数据库中对 AlexNet 模型较高层某个通道具有较强响应的图片块。如图 1-5 所示[37]



​	另一种可视化神经网络的思路是通过特征重构出图像，将重构结果与原图进行比较来分析 CNN 的每一层保留了图像的哪些特征。这种可视化思路将 CNN 提取特征的过程视为编码的过程，而对提取到的特征进行重建的过程正好是编码的逆过程，称为解码过程。2014 年，Aravindh  Mahendran 等人就提出这种通过重构特征的思想来可视化分析CNN[39]。图像理解的关键部分是图像特征表达。而人对图像特征表征的理解是有限的，因此 Mahendran 采用了一种反演的方法来分析图像特征中所含的视觉信息。之后，Mahendran 又加入自然图像先验的信息进一步通过一些可视化技术来加深人们对图像表示的理解。 

​	2015 年，Yosinski[40]根据以往的可视化成果开发出了两个可视化 CNN 模型的工具。其中一个是针对已经训练好的网络，当传入一张图片或一段视频时，通过对该网络中每一层的激活值进行可视化。另一个可视化工具是通过在图像空间加正则化优化来对深度神经网络每一层提取的特征进行可视化。 

​	在通过重建特征可视化 CNN 的基础上，2016 年，Alexey Dosovitskiy 等人通过建立一个上卷积神经网络（Up-Convolutional Neural Networks,UCNN），对 CNN 不同层提取的图像特征进行重建，从而可以知道输入图像中的哪些信息被保留在所提取的特征中[32]。尽管这种方法也能对全连接层进行可视化，也只是显示全连接层保留了哪些信息，而未对这些信息的相关性及重要性进行分析[41]。 

​	周伯雷等人[42]提出的类别激活映射（Class Activation Mapping，CAM）可视化方法，采用 NIN 和 GoogLeNet 中所用到的全局平均池化（Global Average Pooling，GAP），将卷积神经网络最后的全连接层换成全卷积层，并将输出层的权重反向投影至卷积层特征。这一结构的改进能有效定位图像中有助于分类任务的关键区域。从定位的角度来讲，CAM 方法还能起到目标检测的作用，而且不需要给出目标的边框就能大概定位出图像中的目标位置。尽管CAM已经达到了很不错的可视化效果，但要可视化一个通用的CNN模型，就需要用 GAP 层取代最后的全连接层，这就需要修改原模型的结构，导致重新训练该模型带来大量的工作，限制了 CAM 的使用场景。2016 年，R.Selvaraju  等人[43]



在 CAM 的基础上提出了 Grad-CAM。CAM 通过替换全连接层为 GAP 层，重新训练得到权重，而 Grad-CAM 另辟蹊径，用梯度的全局平均来计算每对特征图对应的权重，最后求一个加权和。Grad-CAM 与 CAM 的主要区别在于求权重的过程。



**Deep Visualization:可视化并理解CNN**

https://blog.csdn.net/zchang81/article/details/78095378



**谷歌的新CNN特征可视化方法，构造出一个华丽繁复的新世界** 

https://www.leiphone.com/news/201711/aNw8ZjqMuqvygzlz.html



CNN可视化理解的最新综述

http://m.elecfans.com/article/686276.html



**CNN模型的可视化**

http://www.cctime.com/html/2018-4-12/1373705.htm



CNN特征可视化报告

https://wenku.baidu.com/view/86311603f011f18583d049649b6648d7c1c708ed.html



Visualizing and Understanding Convolutional Networks https://arxiv.org/pdf/1311.2901.pdf



​	深度可视化技术己经成为了深度学习中一个学术研究热点，但仍然处于探索阶段。本文的主要研究对象是深度神经网络中数以千计的卷积滤波器。深度神经网络中不同的滤波器会从输入图像中提取不同特征表示。己有的研究表明低层的卷积核提取了图像的低级语义特性（如边缘、角点），高层的卷积滤波器提取了图像的高层语义特性（如图像类别）。但是，由于深度神经网络会以逐层复合的方式从输入数据中提取特征，我们仍然无法像Ｓｏｂｅｌ算子提取的图像边缘结果图一样直观地观察到深度神经网络中的卷积滤波器从输入图像中提取到的特征表示。



# 第一章 卷积神经网络

https://www.toutiao.com/a6725276071358366215/?tt_from=weixin&utm_campaign=client_share&wxshare_count=1&timestamp=1566823385&app=news_article&utm_source=weixin&utm_medium=toutiao_android&req_id=20190826204304110249203143263D4EED&group_id=6725276071358366215

# 「深度学习系列」CNN模型的可视化

这个网站有cvpr今年的可解释性的文献集合，还挺多的http://openaccess.thecvf.com/CVPR2019_workshops/CVPR2019_Explainable_AI.py、



## 一个很好的可视化网站：http://shixialiu.com/publications/cnnvis/demo/



## 1.1 网络结构

## 1.2 训练过程

## 1.3 网络搭建

## 1.4 优化算法



# 第二章 CNN可视化概述

## 2.5 窥探黑盒-卷积神经网络的可视化**

https://blog.csdn.net/shenziheng1/article/details/85058430

## 目前卷积深度表示的可视化/解释方法

### 中间激活态/特征图可视化。

也就是对卷积神经网络的中间输出特征图进行可视化，这有助于理解卷积神经网络连续的层如何对输入的数据进行展开变化，也有注意了解卷及神经网络每个过滤器的含义。 更深入的， 笔者曾经讲中间激活态结合‘注意力机制’进行联合学习，确实显著提高了算法的精度。	

### 空间滤波器组可视化。

卷积神经网络学习的实质可以简单理解为学习一系列空间滤波器组的参数。可视化滤波器组有助于理解视觉模式/视觉概念。 更深入的，笔者曾经思考过，如何才能引导dropout趋向各项同性空间滤波器。因为从视觉感知对信息的捕捉效果来看，更倾向于捕捉高频成分，诸如边缘特征、纹理等。

### 原始图像中各组分贡献热力图。

我们都知道，卷积神经网络是基于感受野以及感受野再次组合进行特征提取的。但是我们需要了解图像中各个部分对于目标识别的贡献如何？这里将会介绍一种hotmap的形式，判断图像中各个成分对识别结果的贡献度概率。

作者：沈子恒 
来源：CSDN 
原文：https://blog.csdn.net/shenziheng1/article/details/85058430 
版权声明：本文为博主原创文章，转载请附上博文链接！



## 2.1 背景介绍

​	在当前深度学习的领域，有一个非常不好的风气：一切以经验论，好用就行，不问为什么，很少深究问题背后的深层次原因。从长远来看，这样做就埋下了隐患。举个例子，在1980年左右的时候，美国五角大楼启动了一个项目：用神经网络模型来识别坦克(当时还没有深度学习的概念)，他们采集了100张隐藏在树丛中的坦克照片，以及另100张仅有树丛的照片。一组顶尖的研究人员训练了一个神经网络模型来识别这两种不同的场景，这个神经网络模型效果拔群，在测试集上的准确率尽然达到了100%！于是这帮研究人员很高兴的把他们的研究成果带到了某个学术会议上，会议上有个哥们提出了质疑：你们的训练数据是怎么采集的？后来进一步调查发现，原来那100张有坦克的照片都是在阴天拍摄的，而另100张没有坦克的照片是在晴天拍摄的……也就是说，五角大楼花了那么多	的经费，最后就得到了一个用来区分阴天和晴天的分类模型。
当然这个故事应该是虚构的，不过它很形象的说明了什么叫“数据泄露”，这在以前的Kaggle比赛中也曾经出现过。大家不妨思考下，假如我们手里现在有一家医院所有医生和护士的照片，我们希望训练出一个图片分类模型，能够准确的区分出医生和护士。当模型训练完成之后，准确率达到了99%，你认为这个模型可靠不可靠呢？大家可以自己考虑下这个问题。

​	好在学术界的一直有人关注着这个问题，并引申出一个很重要的分支，就是模型的可解释性问题。那么本文从就从近几年来的研究成果出发，谈谈如何让看似黑盒的CNN模型“说话”，对它的分类结果给出一个解释。注意，本文所说的“解释”，与我们日常说的“解释”内涵不一样：例如我们给孩子一张猫的图片，让他解释为什么这是一只猫，孩子会说因为它有尖耳朵、胡须等。而我们让CNN模型解释为什么将这张图片的分类结果为猫，只是让它标出是通过图片的哪些像素作出判断的。（严格来说，这样不能说明模型是否真正学到了我们人类所理解的“特征”，因为模型所学习到的特征本来就和人类的认知有很大区别。何况，即使只标注出是通过哪些像素作出判断就已经有很高价值了，如果标注出的像素集中在地面上，而模型的分类结果是猫，显然这个模型是有问题的）

作者：丽宝儿 
来源：CSDN 
原文：https://blog.csdn.net/heruili/article/details/90214280 
版权声明：本文为博主原创文章，转载请附上博文链接！

## 2.2 反卷积和导向反向传播

​	关于CNN模型的可解释问题，很早就有人开始研究了，姑且称之为CNN可视化吧。比较经典的有两个方法，反卷积(Deconvolution)和导向反向传播(Guided-backpropagation)，通过它们，我们能够一定程度上“看到”CNN模型中较深的卷积层所学习到的一些特征。当然这两个方法也衍生出了其他很多用途，以反卷积为例，它在图像语义分割中有着非常重要的作用。

### 2.2.1 反向传播

### 2.2.2 反卷积

### 2.2.3 导向反向传播

### 2.2.4 反卷积、导向反向传播和反向传播的区别

​	从本质上说，反卷积和导向反向传播的基础都是反向传播，其实说白了就是对输入进行求导，三者唯一的区别在于反向传播过程中经过ReLU层时对梯度的不同处理策略。

![](./img/decon1.png)

![](./img/decon2.png)

![](./img/decon3.png)

![](./img/decon4.png)

作者：丽宝儿 
来源：CSDN 
原文：https://blog.csdn.net/heruili/article/details/90214280 
版权声明：本文为博主原创文章，转载请附上博文链接！



# 第三章 基于反卷积的特征可视化

**使用反卷积（Deconvnet）可视化ＣＮＮ卷积层，查看各层学到的内容**

https://blog.csdn.net/sean2100/article/details/83663212

​	为了解释卷积神经网络为什么work，我们就需要解释CNN的每一层学习到了什么东西。为了理解网络中间的每一层，提取到特征，paper通过反卷积的方法，进行可视化。反卷积网络可以看成是卷积网络的逆过程。反卷积网络在文献《Adaptive deconvolutional networks for mid and high level feature learning》中被提出，是用于无监督学习的。然而本文的反卷积过程并不具备学习的能力，仅仅是用于可视化一个已经训练好的卷积网络模型，没有学习训练的过程。

​	反卷积可视化以各层得到的特征图作为输入，进行反卷积，得到反卷积结果，用以验证显示各层提取到的特征图。举个例子：假如你想要查看Alexnet 的conv5提取到了什么东西，我们就用conv5的特征图后面接一个反卷积网络，然后通过：反池化、反激活、反卷积，这样的一个过程，把本来一张13*13大小的特征图(conv5大小为13*13)，放大回去，最后得到一张与原始输入图片一样大小的图片(227*227)。

## 3.1 反池化过程

 我们知道，池化是不可逆的过程，然而我们可以通过记录池化过程中，最大激活值得坐标位置。然后在反池化的时候，只把池化过程中最大激活值所在的位置坐标的值激活，其它的值置为0，当然这个过程只是一种近似，因为我们在池化的过程中，除了最大值所在的位置，其它的值也是不为0的。刚好最近几天看到文献：《Stacked What-Where Auto-encoders》，里面有个反卷积示意图画的比较好，所有就截下图，用这篇文献的示意图进行讲解：

![](./img/depooling1.png)

以上面的图片为例，上面的图片中左边表示pooling过程，右边表示unpooling过程。假设我们pooling块的大小是3*3，采用max pooling后，我们可以得到一个输出神经元其激活值为9，pooling是一个下采样的过程，本来是3*3大小，经过pooling后，就变成了1*1大小的图片了。而upooling刚好与pooling过程相反，它是一个上采样的过程，是pooling的一个反向运算，当我们由一个神经元要扩展到3*3个神经元的时候，我们需要借助于pooling过程中，记录下最大值所在的位置坐标(0,1)，然后在unpooling过程的时候，就把(0,1)这个像素点的位置填上去，其它的神经元激活值全部为0。再来一个例子：

![](./img/depooling2.png)

在max pooling的时候，我们不仅要得到最大值，同时还要记录下最大值得坐标（-1，-1），然后再unpooling的时候，就直接把(-1-1)这个点的值填上去，其它的激活值全部为0。



## 3.2 反激活

  

我们在Alexnet中，relu函数是用于保证每层输出的激活值都是正数，因此对于反向过程，我们同样需要保证每层的特征图为正值，也就是说这个反激活过程和激活过程没有什么差别，都是直接采用relu函数。

  

## 3.3 反卷积

  

对于反卷积过程，采用卷积过程转置后的滤波器(参数一样，只不过把参数矩阵水平和垂直方向翻转了一下)，反卷积实际上应该叫卷积转置。
 最后可视化网络结构如下：

网络的整个过程，从右边开始：输入图片-》卷积-》Relu-》最大池化-》得到结果特征图-》反池化-》Relu-》反卷积。到了这边，可以说我们的算法已经学习完毕了，其它部分是文献要解释理解CNN部分，可学可不学。



总的来说算法主要有两个关键点：1、反池化  2、反卷积，这两个源码的实现方法，需要好好理解。



## 3.4 特征可视化结果

特征可视化：一旦我们的网络训练完毕了，我们就可以进行可视化，查看学习到了什么东西。但是要怎么看？怎么理解，又是一回事了。我们利用上面的反卷积网络，对每一层的特征图进行查看。

![](E:/CNN-Visualization/img/feature_v1.png)

![](E:/CNN-Visualization/img/feature_v2.png)

总的来说，通过CNN学习后，我们学习到的特征，是具有辨别性的特征，比如要我们区分人脸和狗头，那么通过CNN学习后，背景部位的激活度基本很少，我们通过可视化就可以看到我们提取到的特征忽视了背景，而是把关键的信息给提取出来了。从layer 1、layer 2学习到的特征基本上是颜色、边缘等低层特征；layer 3则开始稍微变得复杂，学习到的是纹理特征，比如上面的一些网格纹理；layer 4学习到的则是比较有区别性的特征，比如狗头；layer 5学习到的则是完整的，具有辨别性关键特征。

## 3.5 特征学习的过程

作者给我们显示了，在网络训练过程中，每一层学习到的特征是怎么变化的，上面每一整张图片是网络的某一层特征图，然后每一行有8个小图片，分别表示网络epochs次数为：1、2、5、10、20、30、40、64的特征图：

  ![](./img/feature_v3.png)

结果：(1)仔细看每一层，在迭代的过程中的变化，出现了sudden jumps;(2)从层与层之间做比较，我们可以看到，低层在训练的过程中基本没啥变化，比较容易收敛，高层的特征学习则变化很大。这解释了低层网络的从训练开始，基本上没有太大的变化，因为梯度弥散嘛。(3)从高层网络conv5的变化过程，我们可以看到，刚开始几次的迭代，基本变化不是很大，但是到了40~50的迭代的时候，变化很大，因此我们以后在训练网络的时候，不要着急看结果，看结果需要保证网络收敛。

# 第四章 基于类别激活映射的特征可视化

## 4.1 GAP

让我们小小地绕行一下，先介绍下**全局平均池化（global average pooling，GAP）**这一概念。为了避免全连接层的过拟合问题，**网中网（Network in Network）**提出了GAP**层**。GAP层，顾名思义，就是对整个特征映射应用平均池化，换句话说，是一种极端激进的平均池化。
作者：论智链接：https://www.zhihu.com/question/274926848/answer/473562723来源：知乎著作权归作者所有。商业转载请联系作者获得授权，非商业转载请注明出处。



## 4.2 CAM

从另一方面来说，GAP层的输出，可以认为是“简要概括”了之前卷积层的特征映射。在网中网架构中，GAP后面接softmax激活，ResNet-50中，GAP层后面接一个带softmax激活的全连接层。softmax激活是为了保证输出分类的概率之和为1，对于热图来说，我们并不需要这一约束。所以可以把softmax拿掉。拿掉softmax的全连接层，其实就是线性回归。结果发现，这样一处理，效果挺不错的：
作者：论智链接：https://www.zhihu.com/question/274926848/answer/473562723来源：知乎著作权归作者所有。商业转载请联系作者获得授权，非商业转载请注明出处。



## 4.3 Grad-CAM

但是CAM要发挥作用，前提是网络架构里面有GAP层，但并不是所有模型都配GAP层的。另外，线性回归的训练是额外的工作。为了克服CAM的这些缺陷，Selvaraju等提出了Grad-CAM。其基本思路是对应于某个分类的特征映射的权重可以表达为梯度，这样就不用额外训练线性回归（或者说线性层）。然后全局平均池化其实是一个简单的运算，并不一定需要专门使用一个网络层。

作者：论智
链接：https://www.zhihu.com/question/274926848/answer/473562723
来源：知乎
著作权归作者所有。商业转载请联系作者获得授权，非商业转载请注明出处。



**基于局部梯度的神经网络可视化解释**

https://www.jianshu.com/p/53062ee77e62



## 4.4 Grad-CAM++

为了得到更好的效果（特别是在某一分类的物体在图像中不止一个的情况下），Chattopadhyay等又进一步提出了Grad-CAM++，主要的变动是在对应于某个分类的特征映射的权重表示中加入了ReLU和权重梯度



## 4.5 CAM、Grad-CAM、Grad-CAM++架构对比

https://blog.csdn.net/weixin_39875161/article/details/90553266



## 4.6 LIME

http://bindog.github.io/blog/2018/02/11/model-explanation-2/



4.1 CAM 

**CAM方法获取显著图：基于pytorch的实现**https://blog.csdn.net/zsx1713366249/article/details/87902476

- [CAM的tensorflow实现]https://github.com/philipperemy/tensorflow-class-activation-mapping
- [Grad-CAM的tensorflow实现]https://github.com/insikk/Grad-CAM-tensorflow

4.2 Grad-CAM

凭什么相信你，我的CNN模型？

http://bindog.github.io/blog/2018/02/10/model-explanation/



4.3 Grad-CAM



2.2.2.2 卷积神经网络可视化——Grad CAM Python实现

https://blog.csdn.net/ZWX2445205419/article/details/86521829



2.2.3 请问注意力机制中生成的类似热力图或者柱状图是如何生成的？

https://www.zhihu.com/question/274926848/answer/473562723



代码

GitHub上有不少Grad-CAM(++)的实现，你可以根据情况自行选择。例如：

- 如果你用TensorFlow，可以看看Hive开源的[hiveml/tensorflow-grad-cam](https://link.zhihu.com/?target=https%3A//github.com/hiveml/tensorflow-grad-cam)
- 如果你用PyTorch，可以看看[jacobgil/pytorch-grad-cam](https://link.zhihu.com/?target=https%3A//github.com/jacobgil/pytorch-grad-cam)

当然，你也可以根据[Grad-CAM++论文](https://link.zhihu.com/?target=https%3A//arxiv.org/abs/1710.11063)自行实现。



2.2.3.5 Guided Grad-CAM(导向反向传播和Grad-CAM的结合)

1）问题：虽然Grad-CAM可以很好的类别判别能力，也可以将相关区域定位出来，但是其不具备像素空间梯度可视化(比如导向反向传播和反卷积这种细粒度重要性可视化)的方法
 　2）解决问题：
 　　2.1）首先对图像使用插值法进行上采样
 　　2.2）然后将导向反向传播和Grad-CAM结合起来，实现可视化



# 第五章 用优化方法形成可视化

**谷歌的新CNN特征可视化方法，构造出一个华丽繁复的新世界** 

https://www.leiphone.com/news/201711/aNw8ZjqMuqvygzlz.html

雷锋网 AI 科技评论按：深度神经网络解释性不好的问题一直是所有研究人员和商业应用方案上方悬着的一团乌云，现代CNN网络固然有强大的特征抽取能力，但没有完善的理论可以描述这个抽取过程的本质，人类也很难理解网络学到的表征。

当然了，研究人员们从来都不会放弃尝试。IMCL 2017的最佳论文奖就颁给了 Pang Wei Koh 和 Percy Liang的「Understanding Black-box Predictions via Influence Functions」，探究训练数据对模型训练过程的影响（现场演讲全文[点这里](https://www.leiphone.com/news/201708/Hbjv7EcuXTYQlLk2.html)）；近期引发全面关注的 Geoffery Hinton的胶囊论文也通过多维激活向量带来了更好的解释性，不同的维度表征着不同的属性（虽然解释性的提高算不上是他的原意；论文全文翻译[看这里](https://www.leiphone.com/news/201710/seYRjGDt30yXcNSr.html)）。

近日，来自谷歌大脑和谷歌研究院的一篇技术文章又从一个新的角度拓展了人类对神经网络的理解，得到的可视化结果也非常亮眼、非常魔性，比如下面这样，文中的结果也在Twitter上引发了许多关注和讨论。

![](./img/opv1.jpg)

这表达的是什么？这又是怎么来的呢？雷锋网 AI 科技评论把研究内容详细介绍如下。

## 5.1 用优化方法形成可视化

作者们的目标是可视化呈现让网络激活的那些特征，也就是回答“模型都在图像中找什么特征”这个问题。他们的思路是新生成一些会让网络激活的图像，而不是看那些数据集中已有的能让网络激活的图像，因为已有图像中的特征很可能只是“有相关性”，在分析的时候可能只不过是“人类从许多特征中选出了自己认为重要的”，而下面的优化方法就能真正找到图像特征和网络行为中的因果性。

总体来说，神经网络是关于输入可微的。如果要找到引发网络某个行为的输入，不管这个行为只是单个神经元的激活还是最终的分类器输出，都可以借助导数迭代地更新输入，最终确认输入图像和选定特征之间的因果关系。（实际执行中当然还有一些技巧，见下文“特征可视化的实现过程”节）

![](./img/opv2.jpg)



从随机噪音开始，迭代优化一张图像让它激活指定的某一个神经元（以4a层的神经元11为例） 

作者们基于带有 Inception 模块的 GoogLeNet展开了研究，这是一个2014年的模型 （https://arxiv.org/pdf/1409.4842.pdf ），当年也以6.67%的前5位错误率拿下了 ILSVRC 2014图像分类比赛的冠军。模型结构示意图如下；训练数据集是 ImageNet。

![](./img/opv3.jpg)

**GoogLeNet 结构示意图。共有9个Inception模块；3a模块前有两组前后连接的卷积层和最大池化层；3b和4a、4e和5a之间各还有一个最大池化层。**

![](./img/opv4.jpg)

## 5.2 优化目标

有了思路和网络之后就要考虑以网络的哪部分结构作为输入优化的目标；即便对于在数据集中找样本的方法也需要考虑这个。这里就有很多种选择，是单个神经元、某个通道、某一层、softmax前的类别值还是softmax之后的概率。不同的选择自然会带来不同的可视化结果，如下图

![](./img/opv5.jpg)

以不同的网络结构为目标可以找到不同的输入图像。这里 n 为层序号，x,y 为空间位置， z 为通道序号，k 为类别序号。 

要理解网络中的单个特征，比如特定位置的某个神经元、或者一整个通道，就可以找让这个特征产生很高的值的样本。文中多数的图像都是以通道作为目标生成的。

要理解网络中的完整一层，就可以用 DeepDream的目标，找到整个层觉得“有兴趣”的图像。

要从分类器的阶段出发找到输入样本的话，会遇到两个选择，优化softmax前的类别值还是优化softmax后的类别概率。softmax前的类别值其实可以看作每个类别出现的证据确凿程度，softmax后的类别概率就是在给定的证据确凿程度之上的似然值。不过不幸的是，增大softmax后的某一类类别概率的最简单的办法不是让增加这一类的概率，而是降低别的类的概率。所以根据作者们的实验，以softmax前的类别值作为优化目标可以带来更高的图像质量。

## 5.3 可视化结果一：不同的层优化得到不同的图像

**3a层**

![](./img/opv6.jpg)

第一个Inception层就已经显示出了一些有意思的纹理。由于每个神经元只有一个很小的感受野，所以整个通道的可视化结果就像是小块纹理反复拼贴的结果。

**3b层**

![](./img/opv7.jpg)

纹理变得复杂了一些，但还都是比较局部的特征

**4a层**

![](./img/opv8.jpg)

4a层跟在了一个最大池化层之后，所以可以看到复杂性大幅度增加。图像中开始出现更复杂的模式，甚至有物体的一部分。

**4b层**

![谷歌的新CNN特征可视化方法，构造出一个华丽繁复的新世界](https://static.leiphone.com/uploads/new/article/740_740/201711/5a0432a3374e4.jpg?imageMogr2/format/jpg/quality/90)

可以明显看到物体的某些部分了，检测台球的例子中就能清楚看到球的样子。这时的可视化结果也开始带有一些环境信息，比如树的例子中就能看到树背后的蓝天和树脚下的地面。

**4c层**

![谷歌的新CNN特征可视化方法，构造出一个华丽繁复的新世界](https://static.leiphone.com/uploads/new/article/740_740/201711/5a0432aeb632e.jpg?imageMogr2/format/jpg/quality/90)

这一层的结果已经足够复杂了，只看几个神经元的优化结果可以比看整个通道更有帮助。有一些神经元只关注拴着的小狗，有的只关注轮子，也有很多其它的有意思的神经元。这也是作者们眼中最有收获的一层。

**4d层**

![谷歌的新CNN特征可视化方法，构造出一个华丽繁复的新世界](https://static.leiphone.com/uploads/new/article/740_740/201711/5a0432b8893ef.jpg?imageMogr2/format/jpg/quality/90)

这一层中有更复杂的概念，比如第一张图里的某种动物的口鼻部分。另一方面，也能看到一些神经元同时对多个没什么关系的概念产生响应。这时需要通过优化结果的多样性和数据集中的样本帮助理解神经元的行为。

**4e层**

![谷歌的新CNN特征可视化方法，构造出一个华丽繁复的新世界](https://static.leiphone.com/uploads/new/article/740_740/201711/5a04349ac6288.jpg?imageMogr2/format/jpg/quality/90)

在这一层，许多神经元已经可以分辨不同的动物种类，或者对多种不同的概念产生响应。不过它们视觉上还是很相似，就会产生对圆盘天线和墨西哥宽边帽都产生反应的滑稽情况。这里也能看得到关注纹理的检测器，不过这时候它们通常对更复杂的纹理感兴趣，比如冰激凌、面包和花椰菜。这里的第一个例子对应的神经元正如大家所想的那样对可以乌龟壳产生反应，不过好玩的是它同样也会对乐器有反应。

**5a层**

![谷歌的新CNN特征可视化方法，构造出一个华丽繁复的新世界](https://static.leiphone.com/uploads/new/article/740_740/201711/5a0432dfc1330.jpg?imageMogr2/format/jpg/quality/90)

这里的可视化结果已经很难解释了，不过它们针对的语义概念都还是比较特定的

**5b层**

![谷歌的新CNN特征可视化方法，构造出一个华丽繁复的新世界](https://static.leiphone.com/uploads/new/article/740_740/201711/5a04348eb4974.jpg?imageMogr2/format/jpg/quality/90)

这层的可视化结果基本都是找不到任何规律的拼贴组合。有可能还能认得出某些东西，但基本都需要多样性的优化结果和数据集中的样本帮忙。这时候能激活神经元的似乎并不是有什么特定语义含义的结构。

## 5.4 可视化结果二：样本的多样性

其实得到可视性结果之后就需要回答一个问题：这些结果就是全部的答案了吗？由于过程中存在一定的随机性和激活的多重性，所以即便这些样本没什么错误，但它们也只展示了特征内涵的某一些方面。

**不同激活程度的样本**

在这里，作者们也拿数据集中的真实图像样本和生成的样本做了比较。真实图像样本不仅可以展现出哪些样本可以极高程度地激活神经元，也能在各种变化的输入中看到神经元分别激活到了哪些程度。如下图

![谷歌的新CNN特征可视化方法，构造出一个华丽繁复的新世界](https://static.leiphone.com/uploads/new/article/740_740/201711/5a056e04a3562.jpg?imageMogr2/format/jpg/quality/90)

![谷歌的新CNN特征可视化方法，构造出一个华丽繁复的新世界](https://static.leiphone.com/uploads/new/article/740_740/201711/5a056e0f78758.jpg?imageMogr2/format/jpg/quality/90)

![谷歌的新CNN特征可视化方法，构造出一个华丽繁复的新世界](https://static.leiphone.com/uploads/new/article/740_740/201711/5a056e1dd56b7.jpg?imageMogr2/format/jpg/quality/90)

可以看到，对真实图像样本来说，多个不同的样本都可以有很高的激活程度。

**多样化样本**

作者们也根据相似性损失或者图像风格转换的方法产生了多样化的样本。如下图

![谷歌的新CNN特征可视化方法，构造出一个华丽繁复的新世界](https://static.leiphone.com/uploads/new/article/740_740/201711/5a056f32226a2.jpg?imageMogr2/format/jpg/quality/90)

多样化的特征可视化结果可以更清晰地看到是哪些结构能够激活神经元，而且可以和数据集中的照片样本做对比，确认研究员们的猜想的正确性（这反过来说就是上文中理解每层网络的优化结果时有时需要依靠多样化的样本和数据集中的样本）。

![谷歌的新CNN特征可视化方法，构造出一个华丽繁复的新世界](https://static.leiphone.com/uploads/new/article/740_740/201711/5a05707f61e98.jpg?imageMogr2/format/jpg/quality/90)

比如这张图中，单独看第一排第一张简单的优化结果，我们很容易会认为神经元激活需要的是“狗头的顶部”这样的特征，因为优化结果中只能看到眼睛和向下弯曲的边缘。在看过第二排的多样化样本之后，就会发现有些样本里没有包含眼睛，有些里包含的是向上弯曲的边缘。这样，我们就需要扩大我们的期待范围，神经元的激活靠的可能主要是皮毛的纹理。带着这个结论再去看看数据集中的样本的话，很大程度上是相符的；可以看到有一张勺子的照片也让神经元激活了，因为它的纹理和颜色都和狗的皮毛很相似。

对更高级别的神经元来说，多种不同类别的物体都可以激活它，优化得到的结果里也就会包含这各种不同的物体。比如下面的图里展示的就是能对多种不同的球类都产生响应的情况。

![谷歌的新CNN特征可视化方法，构造出一个华丽繁复的新世界](https://static.leiphone.com/uploads/new/article/740_740/201711/5a05733c7ab41.jpg?imageMogr2/format/jpg/quality/90)

这种简单的产生多样化样本的方法有几个问题：首先，产生互有区别的样本的压力会在图像中增加无关的瑕疵；而且这个优化过程也会让样本之间以不自然的方式产生区别。比如对于上面这张球的可视化结果，我们人类的期待是看到不同的样本中出现不同种类的球，但实际上更像是在不同的样本中出现了各有不同的特征。

多样性方面的研究也揭露了另一个更基础的问题：上方的结果中展示的都还算是总体上比较相关、比较连续的，也有一些神经元感兴趣的特征是一组奇怪的组合。比如下面图中的情况，这个神经元对两种动物的面容感兴趣，另外还有汽车车身。

![谷歌的新CNN特征可视化方法，构造出一个华丽繁复的新世界](https://static.leiphone.com/uploads/new/article/740_740/201711/5a057645796c6.jpg?imageMogr2/format/jpg/quality/90)

类似这样的例子表明，想要理解神经网络中的表义过程时，神经元可能不一定是合适的研究对象。

## 5.5 可视化结果三：神经元间的互动 

如果神经元不是理解神经网络的正确方式，那什么才是呢？作者们也尝试了神经元的组合。实际操作经验中，我们也认为是一组神经元的组合共同表征了一张图像。单个神经元就可以看作激活空间中的单个基础维度，目前也没发现证据证明它们之间有主次之分。

作者们尝试了给神经元做加减法，比如把表示“黑白”的神经元加上一个“马赛克”神经元，优化结果就是同一种马赛克的黑白版本。这让人想起了Word2Vec中词嵌入的语义加减法，或者生成式模型中隐空间的加减法。

联合优化两个神经元，可以得到这样的结果。

![谷歌的新CNN特征可视化方法，构造出一个华丽繁复的新世界](https://static.leiphone.com/uploads/new/article/740_740/201711/5a057a63acc47.jpg?imageMogr2/format/jpg/quality/90)

![谷歌的新CNN特征可视化方法，构造出一个华丽繁复的新世界](https://static.leiphone.com/uploads/new/article/740_740/201711/5a057a636526f.jpg?imageMogr2/format/jpg/quality/90)

也可以在两个神经元之间取插值，便于更好理解神经元间的互动。这也和生成式模型的隐空间插值相似。

![谷歌的新CNN特征可视化方法，构造出一个华丽繁复的新世界](https://static.leiphone.com/uploads/new/article/740_740/201711/5a057b7195674.jpg?imageMogr2/format/jpg/quality/90)

![谷歌的新CNN特征可视化方法，构造出一个华丽繁复的新世界](https://static.leiphone.com/uploads/new/article/740_740/201711/5a057b7186c65.jpg?imageMogr2/format/jpg/quality/90)

不过这些也仅仅是神经元间互动关系的一点点皮毛。实际上作者们也根本不知道如何在特征空间中选出有意义的方向，甚至都不知道到底有没有什么方向是带有具体的含义的。除了找到方向之外，不同反向之间如何互动也还存在疑问，比如刚才的差值图展示出了寥寥几个神经元之间的互动关系，但实际情况是往往有数百个神经元、数百个方向之间互相影响。

## 5.6 特征可视化的实现过程

如前文所说，作者们此次使用的优化方法的思路很简单，但想要真的产生符合人类观察习惯的图像就需要很多的技巧和尝试了。直接对图像进行优化可能会产生一种神经网络的光学幻觉 —— 人眼看来是一副全是噪声、带有看不出意义的高频图样的图像，但网络却会有强烈的响应。即便仔细调整学习率，还是会得到明显的噪声。（下图学习率0.05）

![谷歌的新CNN特征可视化方法，构造出一个华丽繁复的新世界](https://static.leiphone.com/uploads/new/article/740_740/201711/5a057e0e32e83.jpg?imageMogr2/format/jpg/quality/90)

这些图样就像是作弊图形，用现实生活中不存在的方式激活了神经元。如果优化的步骤足够多，最终得到的东西是神经元确实有响应，但人眼看来全都是高频图样的图像。这种图样似乎和对抗性样本的现象之间有紧密的关系。（雷锋网(公众号：雷锋网) AI 科技评论编译也有同感，关于对抗性样本的更早文章可以看[这里](https://www.leiphone.com/news/201706/CrSyyhCUNz2gYIIJ.html)）

作者们也不清楚这些高频图样的具体产生原因，他们猜想可能和带有步幅的卷积和最大池化操作有关系，两者都可以在梯度中产生高频率的图样。

![谷歌的新CNN特征可视化方法，构造出一个华丽繁复的新世界](https://static.leiphone.com/uploads/new/article/740_740/201711/5a0582182ea94.jpg?imageMogr2/format/jpg/quality/90)

通过反向传播过程作者们发现，每次带有步幅的卷积或者最大池化都会在梯度维度中产生棋盘般的图样

这些高频图样说明，虽然基于优化方法的可视化方法不再受限于真实样本，有着极高的自由性，它却也是一把双刃剑。如果不对图像做任何限制，最后得到的就是对抗性样本。这个现象确实很有意思，但是作者们为了达到可视化的目标，就需要想办法克服这个现象。

## 5.7 不同规范化方案的对比

在特征可视化的研究中，高频噪音一直以来都是主要的难点和重点攻关方向。如果想要得到有用的可视化结果，就需要通过某些先验知识、规范化或者添加限制来产生更自然的图像结构。

实际上，如果看看特征可视化方面最著名的论文，它们最主要的观点之一通常都是使用某种规范化方法。不同的研究者们尝试了许多不同的方法。

文章作者们根据对模型的规范化强度把所有这些方法看作一个连续的分布。在分布的一端，是完全不做规范化，得到的结果就是对抗性样本；在另一端则是在现有数据集中做搜索，那么会出现的问题在开头也就讲过了。在两者之间就有主要的三大类规范化方法可供选择。

![谷歌的新CNN特征可视化方法，构造出一个华丽繁复的新世界](https://static.leiphone.com/uploads/new/article/740_740/201711/5a05909dd294a.jpg?imageMogr2/format/jpg/quality/90)

**频率惩罚**直接针对的就是高频噪音。它可以显式地惩罚相邻像素间出现的高变化，或者在每步图像优化之后增加模糊，隐式地惩罚了高频噪音。然而不幸的是，这些方法同时也会限制合理的高频特征，比如噪音周围的边缘。如果增加一个双边过滤器，把边缘保留下来的话可以得到一些改善。如下图。

![谷歌的新CNN特征可视化方法，构造出一个华丽繁复的新世界](https://static.leiphone.com/uploads/new/article/740_740/201711/5a0592ae99e02.jpg?imageMogr2/format/jpg/quality/90)

**变换健壮性**会尝试寻找那些经过小的变换以后仍然能让优化目标激活的样本。对于图像的例子来说，细微的一点点变化都可以起到明显的作用，尤其是配合使用一个更通用的高频规范器之后。具体来说，这就代表着可以随机对图像做抖动、宣传或者缩放，然后把它应用到优化步骤中。如下图。

![谷歌的新CNN特征可视化方法，构造出一个华丽繁复的新世界](https://static.leiphone.com/uploads/new/article/740_740/201711/5a0592c1ad003.jpg?imageMogr2/format/jpg/quality/90)

**先验知识**。作者们一开始使用的规范化方法都只用到了非常简单的启发式方法来保持样本的合理性。更自然的做法是从真实数据学出一个模型，让这个模型迫使生成的样本变得合理。如果有一个强力的模型，得到的效果就会跟搜索整个数据集类似。这种方法可以得到最真实的可视化结果，但是就很难判断结果中的哪些部分来自研究的模型本身的可视化，哪些部分来自后来学到的模型中的先验知识。

有一类做法大家都很熟悉了，就是学习一个生成器，让它的输出位于现有数据样本的隐空间中，然后在这个隐空间中做优化。比如GAN或者VAE。也有个替代方案是学习一种先验知识，通过它控制概率梯度；这样就可以让先验知识和优化目标共同优化。为先验知识和类别的可能性做优化是，就同步形成了一个限制在这个特定类别数据下的生成式模型。

## 5.8 预处理与参数化

前面介绍的几种方法都降低了梯度中的高频成分，而不是直接去除可视化效果中的高频；它们仍然允许高频梯度形成，只不过随后去减弱它。

有没有办法不让梯度产生高频呢？这里就有一个强大的梯度变换工具：优化中的“预处理”。可以把它看作同一个优化目标的最速下降法，但是要在这个空间的另一个参数化形式下进行，或者在另一种距离下进行。这会改变最快速的那个下降方向，以及改变每个方向中的优化速度有多快，但它并不会改变最小值。如果有许多局部极小值，它还可以拉伸、缩小它们的范围大小，改变优化过程会掉入哪些、不掉入哪些。最终的结果就是，如果用了正确的预处理方法，就可以让优化问题大大简化。

那么带有这些好处的预处理器如何选择呢？首先很容易想到的就是让数据去相关以及白化的方法。对图像来说，这就意味着以Fourier变换做梯度下降，同时要缩放频率的大小这样它们可以都具有同样的能量。

不同的距离衡量方法也会改变最速下降的方向。L2范数梯度就和L∞度量或者去相关空间下的方向很不一样。

![谷歌的新CNN特征可视化方法，构造出一个华丽繁复的新世界](https://static.leiphone.com/uploads/new/article/740_740/201711/5a0599e74a676.jpg?imageMogr2/format/jpg/quality/90)

所有这些方向都是同一个优化目标下的可选下降方向，但是视觉上看来它们的区别非常大。可以看到在去相关空间中做优化能够减少高频成分的出现，用L∞则会增加高频。

选用去相关的下降方向带来的可视化结果也很不一样。由于超参数的不同很难做客观的比较，但是得到的结果看起来要好很多，而且形成得也要快得多。这篇文章中的多数图片就都是用去相关空间的下降和变换健壮性方法一起生成的（除特殊标明的外）。

![谷歌的新CNN特征可视化方法，构造出一个华丽繁复的新世界](https://static.leiphone.com/uploads/new/article/740_740/201711/5a059b0767f3a.jpg?imageMogr2/format/jpg/quality/90)

![谷歌的新CNN特征可视化方法，构造出一个华丽繁复的新世界](https://static.leiphone.com/uploads/new/article/740_740/201711/5a059b077d476.jpg?imageMogr2/format/jpg/quality/90)

![谷歌的新CNN特征可视化方法，构造出一个华丽繁复的新世界](https://static.leiphone.com/uploads/new/article/740_740/201711/5a059b0770e9b.jpg?imageMogr2/format/jpg/quality/90)

那么，是不是不同的方法其实都能下降到同一个点上，是不是只不过正常的梯度下降慢一点、预处理方法仅仅加速了这个下降过程呢？还是说预处理方法其实也规范化（改变）了最终达到的局部极小值点？目前还很难说得清。一方面，梯度下降似乎能一直让优化过程进行下去，只要增加优化过程的步数 —— 它往往并没有真的收敛，只是在非常非常慢地移动。另一方面，如果关掉所有其它的规范化方法的话，预处理方法似乎也确实能减少高频图案的出现。

## 5.9 结论 

文章作者们提出了一种新的方法创造令人眼前一亮的可视化结果，在呈现了丰富的可视化结果同时，也讨论了其中的重大难点和如何尝试解决它们。

在尝试提高神经网络可解释性的漫漫旅途中，特征可视化是最有潜力、得到了最多研究的方向之一。不过单独来看，特征可视化也永远都无法带来完全让人满意的解释。作者们把它看作这个方向的基础研究之一，虽然现在还有许多未能解释的问题，但我们也共同希望在未来更多工具的帮助下，人们能够真正地理解深度学习系统。







# 第六章 理解与可视化卷积神经网络

## [12.1 可视化卷积神经网络学习到的东西](https://blog.csdn.net/vvcrm01/article/details/82110877#12.1 可视化卷积神经网络学习到的东西)

  

## [12.1.1可视化激活和第一层权重](https://blog.csdn.net/vvcrm01/article/details/82110877#12.1.1可视化激活和第一层权重)

  

## [12.1.2 找到对神经元有最大激活的图像](https://blog.csdn.net/vvcrm01/article/details/82110877#retrieving-images-that-maximally-activate-a-neuron)

  

## [12.1.3 用 t-SNE 嵌入代码](https://blog.csdn.net/vvcrm01/article/details/82110877#12.1.3 用 t-SNE 嵌入代码)

  

## [12.1.4 遮挡部分图像](https://blog.csdn.net/vvcrm01/article/details/82110877#12.1.4 遮挡部分图像)

  

## [12.1.5 可视化数据梯度及其他文献](https://blog.csdn.net/vvcrm01/article/details/82110877#12.1.5 可视化数据梯度及其他文献)

  

## [12.1.6 基于CNN代码重构原始图像](https://blog.csdn.net/vvcrm01/article/details/82110877#12.1.6 基于CNN代码重构原始图像)

  

## [12.1.7 保存了多少空间信息？](https://blog.csdn.net/vvcrm01/article/details/82110877#12.1.7 保存了多少空间信息？)

  

## [12.1.8 根据图像属性绘制性能](https://blog.csdn.net/vvcrm01/article/details/82110877#12.1.8 根据图像属性绘制性能)

  

## [12.2 玩弄 ConvNets](https://blog.csdn.net/vvcrm01/article/details/82110877#12.2 玩弄 ConvNets)

  

## [12.3 将ConvNets 的结果与人类标签比较](https://blog.csdn.net/vvcrm01/article/details/82110877#12.3 将ConvNets 的结果与人类标签比较)





# 第七章 可视化工具

https://blog.csdn.net/dcxhun3/article/details/77746550



## 5.11.4 常见的网络可视化方法

​	Tensorflow，Pytorch等每一个主流的深度学习框架都提供了相对应的可视化模板，那有没有一种方法更加具有通用性呢？下面介绍常见的网络可视化方法：

### （1）Netron。

Netron支持主流各种框架的模型结构可视化工作，github链接：https://github.com/lutzroeder/Netron 。

### （2）Netscope。

Netscope在线可视化链接：http://ethereon.github.io/netscope/#/editor。

### （2）ConvNetDraw。

ConvNetDraw的github链接：https://github.com/cbovar/ConvNetDraw。

### （3）Draw_convnet。

Draw_convnet的github链接：https://github.com/gwding/draw_convnet。

### （4）PlotNeuralNet。

PlotNeuralNet的github链接：https://github.com/HarisIqbal88/PlotNeuralNet。

### （5）NN-SVG。

NN-SVG的github链接：https://github.com/zfrenchee/NN-SVG。

### （6）Python + Graphviz。

针对节点较多的网络，用python编写一个简单的dot脚本生成工具（MakeNN），可以很方便的输入参数生成nn结构图。

### （7）Graphviz - dot。

Graphviz的官方链接：https://www.graphviz.org/。

### （8）NetworkX。

NetworkX的github链接：https://github.com/networkx。

### （9）DAFT。

daft官网链接：http://daft-pgm.org/。





# 第三章 可视化组件

AlexNet进行了可视化

介绍三种可视化方法 

## 卷积核输出的可视化

卷积核输出的可视化(Visualizing intermediate convnet outputs (intermediate activations)，即可视化卷积核经过激活之后的结果。能够看到图像经过卷积之后结果，帮助理解卷积核的作用





## 卷积核的可视化



卷积核的可视化(Visualizing convnets filters)，帮助我们理解卷积核是如何感受图像的





## 热度图可视化



热度图可视化(Visualizing heatmaps of class activation in an image)，通过热度图，了解图像分类问题中图像哪些部分起到了关键作用，同时可以定位图像中物体的位置。

作者：芥末的无奈 
来源：CSDN 
原文：https://blog.csdn.net/weiwei9363/article/details/79112872 
版权声明：本文为博主原创文章，转载请附上博文链接！





# 第八章 可视化教程

## 8.1 基于DeepStream的CNN的可视化理解

https://blog.csdn.net/sparkexpert/article/details/74529094



## 8.2 Tensorflow实现卷积特征的可视化

*简单卷积神经网络的tensorboard可视化**

https://blog.csdn.net/happyhorizion/article/details/77894048



https://blog.csdn.net/u014281392/article/details/74316028



## 8.3 基于MatConvNet框架的CNN卷积层与特征图可视化

https://blog.csdn.net/jsgaobiao/article/details/80361494

程序下载链接：https://download.csdn.net/download/jsgaobiao/10422273 
VGG-f模型链接：http://www.vlfeat.org/matconvnet/models/imagenet-vgg-f.mat

   【题目】 
编程实现可视化卷积神经网络的特征图，并探究图像变换(平移，旋转，缩放等）对特征图的影响。选择AlexNet等经典CNN网络的Pre-trained模型，可视化每个卷积层的特征图（网络输入图片自行选择）。其中，第一层全部可视化，其余层选取部分特征图进行可视化。然后对图像进行变换，观察图像变换后特征图的变化。

 【方法概述】 
本次实验使用了VGG-f作为预先加载的模型，通过MATLAB中的load方法将imagenet-vgg-f中的参数加载进程序。 
imagenet-vgg-f是一个21层的卷积神经网络，其参数在ImageNet数据集上进行了训练。它的网络结构包括了5层卷积层、3层全连接层，输出的类别可达1000种。网络结构图太长了放在文章最后。

实验中共有6个输入图像，分别是原图input.jpg以及对它进行平移、缩放、旋转、水平翻转、垂直翻转后的图像 

首先将输入图像进行归一化操作，也就是将图片resize到网络的标准输入大小224*224，并且将图片的每个像素与均值图片的每个像素相减，再输入网络。 
接下来，可视化卷积核的时候，将网络第一层卷积核的参数net.layers{1}.weights{1}提取出来，并使用vl_imarraysc函数进行可视化。第一层卷积核的3个通道在可视化的过程中就被当作RGB三个通道。 
对于feature map的可视化任务，需要先使用vl_simplenn将图片输入神经网络并获取其输出结果。我们需要可视化的是每个卷积层后经过ReLU的结果，每个输入图像对应5个特征图。

 【结果分析】 
由于卷积核的参数是预训练得到的，与输入图片无关，所以只展现一幅图就够了。如下图所示，第一层卷积核学到了图片中一些基础性的特征，比如各种方向的边缘和角点。 

下面展示的是原始图片输入后，5个卷积层的可视化结果。需要说明的是，第二层之后的特征图数量较多，因此每层只选取了64个进行可视化。另外，特征图是单通道的灰度图片，为了可视化的效果更好，我将灰度值映射到了“蓝-黄”的颜色区间内，进行了伪彩色的处理，得到了如下的可视化结果。

其中，第一层特征图的细节比较清晰和输入图片较为相似，提取出了输入图片的边缘（包括刺猬身上的刺）。第2、3、4层特征图的分辨率较低，已经难以看出输入图片的特征，但是可以发现有些特征图对背刺区域激活显著，有些特征图对刺猬的外轮廓、背景等区域激活显著。可以猜测，它们提取了图片中比边缘和角点更高层的语义信息。最后一层特征图中有少量对背刺区域激活显著，少量几乎没有被激活。可以猜测，刺猬的背刺特征是网络判断其类别的显著特征，因此被分类为刺猬的图片在最后一个特征层的背刺区域激活最为明显。 



 【对比分析】 
由于篇幅限制，这里只放置较小的略缩图，高清图片可以运行程序自行查看。

我们先对比最清晰的第一层特征图的可视化结果。 
可以看出除了缩放的图片以外，其他特征图都随着输入图片的变化而变化：平移的图片作为输入，特征图也产生了相对的平移；翻转、旋转都有类似的效果。只有缩放的输入图片并不影响特征图的表现，其原因应该是VGG-f采用固定大小的输入数据，因此不论图片是否经过缩放，在输入VGG-f之前都会被归一化为同样的大小，所以直观上看并不影响特征图的表现。但是由于分辨率的不同，经过resize之后的图片可能会有像素级别的细微差异，人眼不容易分辨出来。

从另一方面来说，虽然特征图对于输入图片的变换产生了相同的变换，但是特征图中的激活区域并没有显著的变化。这说明VGG-f在图片分类的任务中，对输入图片的大小、旋转、翻转、平移等变化是不敏感的，并不会显著影响其分类结果的准确性。也说明了CNN网络具有一定程度的旋转/平移不变性。 


与第一层特征图类似，其他层的特征图也产生了类似的表现，即除了缩放的图片以外，其他作用于输入图片的变换均体现在了特征图上。由于篇幅所限，这里不再单独放出。运行程序即可得到结果。

附上程序下载链接：https://download.csdn.net/download/jsgaobiao/10422273



[VGG-f网络结构图]

作者：jsgaobiao 
来源：CSDN 
原文：https://blog.csdn.net/jsgaobiao/article/details/80361494 
版权声明：本文为博主原创文章，转载请附上博文链接！



## 8.4 Caffe 特征图可视化

https://blog.csdn.net/u012938704/article/details/52767695

## 8.5 基于Keras的CNN可视化

https://blog.csdn.net/weiwei9363/article/details/79112872

1. [deep-learning-with-python-notebooks](https://nbviewer.jupyter.org/github/fchollet/deep-learning-with-python-notebooks/tree/master)
2. 
3. [5.4-visualizing-what-convnets-learn.ipynb](https://nbviewer.jupyter.org/github/fchollet/deep-learning-with-python-notebooks/tree/master/5.4-visualizing-what-convnets-learn.ipynb)

In [1]:

```
import keras
keras.__version__
```



```
Using TensorFlow backend.
```

Out[1]:

```
'2.0.8'
```



# Visualizing what convnets learn

This notebook contains the code sample found in Chapter 5, Section 4 of [Deep Learning with Python](https://www.manning.com/books/deep-learning-with-python?a_aid=keras&a_bid=76564dff). Note that the original text features far more content, in particular further explanations and figures: in this notebook, you will only find source code and related comments.

------

It is often said that deep learning models are "black boxes", learning representations that are difficult to extract and present in a human-readable form. While this is partially true for certain types of deep learning models, it is definitely not true for convnets. The representations learned by convnets are highly amenable to visualization, in large part because they are *representations of visual concepts*. Since 2013, a wide array of techniques have been developed for visualizing and interpreting these representations. We won't survey all of them, but we will cover three of the most accessible and useful ones:

- Visualizing intermediate convnet outputs ("intermediate activations"). This is useful to understand how successive convnet layers transform their input, and to get a first idea of the meaning of individual convnet filters.
- Visualizing convnets filters. This is useful to understand precisely what visual pattern or concept each filter in a convnet is receptive to.
- Visualizing heatmaps of class activation in an image. This is useful to understand which part of an image where identified as belonging to a given class, and thus allows to localize objects in images.

For the first method -- activation visualization -- we will use the small convnet that we trained from scratch on the cat vs. dog classification problem two sections ago. For the next two methods, we will use the VGG16 model that we introduced in the previous section.



## Visualizing intermediate activations

Visualizing intermediate activations consists in displaying the feature maps that are output by various convolution and pooling layers in a network, given a certain input (the output of a layer is often called its "activation", the output of the activation function). This gives a view into how an input is decomposed unto the different filters learned by the network. These feature maps we want to visualize have 3 dimensions: width, height, and depth (channels). Each channel encodes relatively independent features, so the proper way to visualize these feature maps is by independently plotting the contents of every channel, as a 2D image. Let's start by loading the model that we saved in section 5.2:

In [2]:

```
from keras.models import load_model

model = load_model('cats_and_dogs_small_2.h5')
model.summary()  # As a reminder.
```



```
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
conv2d_5 (Conv2D)            (None, 148, 148, 32)      896       
_________________________________________________________________
max_pooling2d_5 (MaxPooling2 (None, 74, 74, 32)        0         
_________________________________________________________________
conv2d_6 (Conv2D)            (None, 72, 72, 64)        18496     
_________________________________________________________________
max_pooling2d_6 (MaxPooling2 (None, 36, 36, 64)        0         
_________________________________________________________________
conv2d_7 (Conv2D)            (None, 34, 34, 128)       73856     
_________________________________________________________________
max_pooling2d_7 (MaxPooling2 (None, 17, 17, 128)       0         
_________________________________________________________________
conv2d_8 (Conv2D)            (None, 15, 15, 128)       147584    
_________________________________________________________________
max_pooling2d_8 (MaxPooling2 (None, 7, 7, 128)         0         
_________________________________________________________________
flatten_2 (Flatten)          (None, 6272)              0         
_________________________________________________________________
dropout_1 (Dropout)          (None, 6272)              0         
_________________________________________________________________
dense_3 (Dense)              (None, 512)               3211776   
_________________________________________________________________
dense_4 (Dense)              (None, 1)                 513       
=================================================================
Total params: 3,453,121
Trainable params: 3,453,121
Non-trainable params: 0
_________________________________________________________________
```



This will be the input image we will use -- a picture of a cat, not part of images that the network was trained on:

In [3]:

```
img_path = '/Users/fchollet/Downloads/cats_and_dogs_small/test/cats/cat.1700.jpg'

# We preprocess the image into a 4D tensor
from keras.preprocessing import image
import numpy as np

img = image.load_img(img_path, target_size=(150, 150))
img_tensor = image.img_to_array(img)
img_tensor = np.expand_dims(img_tensor, axis=0)
# Remember that the model was trained on inputs
# that were preprocessed in the following way:
img_tensor /= 255.

# Its shape is (1, 150, 150, 3)
print(img_tensor.shape)
```



```
(1, 150, 150, 3)
```



Let's display our picture:

In [4]:

```
import matplotlib.pyplot as plt

plt.imshow(img_tensor[0])
plt.show()
```



![img](data:image/png;base64,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%0A2tZ2zv4/OfB7sq8RpuIAAAAASUVORK5CYII=)



In order to extract the feature maps we want to look at, we will create a Keras model that takes batches of images as input, and outputs the activations of all convolution and pooling layers. To do this, we will use the Keras class `Model`. A `Model` is instantiated using two arguments: an input tensor (or list of input tensors), and an output tensor (or list of output tensors). The resulting class is a Keras model, just like the `Sequential` models that you are familiar with, mapping the specified inputs to the specified outputs. What sets the `Model` class apart is that it allows for models with multiple outputs, unlike `Sequential`. For more information about the `Model` class, see Chapter 7, Section 1.

In [5]:

```
from keras import models

# Extracts the outputs of the top 8 layers:
layer_outputs = [layer.output for layer in model.layers[:8]]
# Creates a model that will return these outputs, given the model input:
activation_model = models.Model(inputs=model.input, outputs=layer_outputs)
```



When fed an image input, this model returns the values of the layer activations in the original model. This is the first time you encounter a multi-output model in this book: until now the models you have seen only had exactly one input and one output. In the general case, a model could have any number of inputs and outputs. This one has one input and 8 outputs, one output per layer activation.

In [6]:

```
# This will return a list of 5 Numpy arrays:
# one array per layer activation
activations = activation_model.predict(img_tensor)
```



For instance, this is the activation of the first convolution layer for our cat image input:

In [7]:

```
first_layer_activation = activations[0]
print(first_layer_activation.shape)
```



```
(1, 148, 148, 32)
```



It's a 148x148 feature map with 32 channels. Let's try visualizing the 3rd channel:

In [8]:

```
import matplotlib.pyplot as plt

plt.matshow(first_layer_activation[0, :, :, 3], cmap='viridis')
plt.show()
```



![img](data:image/png;base64,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%0AAT53pdymht3xAAAAAElFTkSuQmCC)



This channel appears to encode a diagonal edge detector. Let's try the 30th channel -- but note that your own channels may vary, since the specific filters learned by convolution layers are not deterministic.

In [9]:

```
plt.matshow(first_layer_activation[0, :, :, 30], cmap='viridis')
plt.show()
```



![img](data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAQsAAAECCAYAAADpWvKaAAAABHNCSVQICAgIfAhkiAAAAAlwSFlz%0AAAALEgAACxIB0t1+/AAAFtJJREFUeJzt3XuQXGWZx/Hv092ZTAK5JyQhk5DBRFJBAoQAIbgq4spl%0AKbJbpSxKrVGxUq7sikqVEqly3ap1V1YXxV2VZQXBLUQQcU2xKEIMWpYymgQIJCQk5johk/sNsrl0%0Az7N/nDOhcyPvdJ/Tp6f796mamtPnnJn3eWd6nnnfc3vM3REROZVc1gGISN+gZCEiQZQsRCSIkoWI%0ABFGyEJEgShYiEiTzZGFmV5vZSjNbbWa3p9zWeDNbaGbLzWyZmd0arx9uZk+b2ar487AU2s6b2fNm%0A9kT8ut3MOuJ+P2JmLUm3Gbcz1MweM7MVZvaKmV1Wo/5+Nv4Zv2xmD5tZaxp9NrP7zWyrmb1ctu6E%0A/bPIt+L2l5rZ9BTa/lr8s15qZj81s6Fl2+bFba80s6uSbLds221m5mY2Mn6dXJ/dPbMPIA/8CTgb%0AaAFeBKam2N5YYHq8PAh4FZgK/Ctwe7z+duDOFNr+HPBD4In49aPAjfHyPcDfptTnB4FPxMstwNC0%0A+wuMA9YCA8r6+tE0+gy8C5gOvFy27oT9A64Ffg4YMBPoSKHt9wOFePnOsranxu/v/kB7/L7PJ9Vu%0AvH488BSwHhiZdJ8Tf3P2stOXAU+VvZ4HzKth+z8D/hxYCYyN140FVibcThuwAHgv8ET8i9te9qY6%0A6ueQYLtD4j9aO2Z92v0dB2wEhgOFuM9XpdVnYOIxf7An7B/wn8CHTrRfUm0fs+2vgIfi5aPe2/Ef%0A9WVJtgs8BpwPrCtLFon1OetpSM+bqkdnvC51ZjYRuBDoAEa7++Z4UxcwOuHmvgl8HuiOX48Adrt7%0AMX6dVr/bgW3A9+Mp0PfM7DRS7q+7bwK+DmwANgN7gMXUps9w8v7V+v32caL/6qm3bWazgU3u/uIx%0AmxJrN+tkkQkzOx34CfAZd99bvs2j9JvYNfBmdh2w1d0XJ/U9e6FANFz9rrtfCLxBNCw/Iun+AsTH%0ACGYTJaszgdOAq5NsI1Qa/QthZncAReChGrQ1EPgi8KU028k6WWwimmf1aIvXpcbM+hEliofc/fF4%0A9RYzGxtvHwtsTbDJy4HrzWwd8COiqcjdwFAzK8T7pNXvTqDT3Tvi148RJY80+wvwPmCtu29z98PA%0A40Q/h1r0GU7ev5q838zso8B1wE1xskq77bcRJeYX4/dZG7DEzMYk2W7WyeKPwOT4KHkLcCMwP63G%0AzMyA+4BX3P2usk3zgTnx8hyiYxmJcPd57t7m7hOJ+vcrd78JWAh8II02y9ruAjaa2TnxqiuB5aTY%0A39gGYKaZDYx/5j3tpt7n2Mn6Nx/4SHyGYCawp2y6kggzu5poynm9u+8/JqYbzay/mbUDk4E/JNGm%0Au7/k7me4+8T4fdZJdCC/iyT7nMQBpioPEF1LdFbiT8AdKbf1TqIh6VLghfjjWqJjCAuAVcAzwPCU%0A2n8Pb54NOZvozbIa+DHQP6U2LwAWxX3+H2BYLfoL/COwAngZ+G+iswCJ9xl4mOi4yOH4j+Tmk/WP%0A6MDyt+P32kvAjBTaXk10jKDn/XVP2f53xG2vBK5Jst1jtq/jzQOcifXZ4m8oIvKWsp6GiEgfoWQh%0AIkGULEQkiJKFiARRshCRIKklC+vl3aRmNjetWNRu9u1m2bbaTUYqycLM8kTndq8hutvuQ2Y29RRf%0AltWbWO02fttqNwFpjSwuAVa7+xp3P0R0mfPslNoSkRoonHqXipzoTrdLT7Zzi/X3VgYy2IbX/Aox%0Atdv4bavdkzvAGxzygxayb1rJ4pTiedVciDr3Trs2q1BEmlaHLwjeN61pyCnvdHP3e919hrvP6Ef/%0AlMIQkaSklSxqejepiKQvlWmIuxfN7O+IHh2WB+5392VptCUitZHaMQt3fxJ4Mq3vLyK1pSs4RSSI%0AkoWIBFGyEJEgShYiEkTJQkSCKFmISBAlCxEJomQhIkEaNlkcuvpiDl19cdZhiDSMhk0WIpKszG5R%0AT1vLL/6YdQgiDUUjCxEJomQhIkGULEQkiJKFiARRshCRIEoWIhJEyUKkRnLTppCbNiXrMCqmZCEi%0AQRr2oiyRetO9dEXWIVRFIwsRCVJxsjCz8Wa20MyWm9kyM7s1Xj/czJ42s1Xx52HJhSsiWalmZFEE%0AbnP3qcBM4Ja4UvrtwAJ3nwwsiF+LNJ3SFdOzDiFRFScLd9/s7kvi5X3AK0QFkWcDD8a7PQj8ZbVB%0AivQl+aFDoo+FS7IOJVGJHLMws4nAhUAHMNrdN8ebuoDRSbQhItmq+myImZ0O/AT4jLvvNXuzeru7%0Au5mdsPT7sVXURRpFafeerENIRVUjCzPrR5QoHnL3x+PVW8xsbLx9LLD1RF+rKuoifUs1Z0MMuA94%0Axd3vKts0H5gTL88BflZ5eCJSL6qZhlwO/A3wkpm9EK/7IvBV4FEzuxlYD9xQXYgi9Sk/ahQApW3b%0AMo6kNipOFu7+W8BOsvnKSr+vSD0rjG+LFtwpdm7KNpga0xWcIhJE94aI9EJxY2fWIWRGIwuRQBu+%0APIvCmNEUxjTnpUNKFiISRNMQkVOZOQ2As7+/kWLXloyDyY6ShchJ2EXnAuDPLQWiOyebmaYhIhJE%0AIwuRk/DFy7IOoa5oZCF9hvVrSb+Ni8/DLj4v9Xb6IiULEQmiZCF9hh8+lOr3L4xvI79tT/QR3/ch%0Ab9IxC5EexSJ0R/8/m+XmsN7QyEJEgmhkIU3v8PsuihaeWZxtIHVOySIj+cGDKe3dm3UYAvRTkgii%0AaYiIBNHIIiMaVWQnP2oUdnr0kOji2vUZR9N3KFlkKD9iOADW2kpx02sZR9M8Stu2gU529JqmISIS%0ARCOLDJV27Mw6BJFgGlmISJCqk4WZ5c3seTN7In7dbmYdZrbazB4xs/Tv/hE5hVxr65HlnmNF0jtJ%0AjCxuJSqK3ONO4BvuPgnYBdycQBsivZYbOJD8OZPInzOJ7gMHjqzX9K8y1ZYvbAP+Avhe/NqA9wKP%0AxbuoirpIg6h2ZPFN4PNAd/x6BLDb3XueQNYJjKuyDZFguQumghmY0T1tEqWVqymtXJ11WA2hmlqn%0A1wFb3b2ia2XNbK6ZLTKzRYc5WGkYIkfpfmE5uEcf8bMzJRnV1jq93syuBVqBwcDdwFAzK8Sjizbg%0AhDXe3P1e4F6AwTbcq4hDRGqg4pGFu89z9zZ3nwjcCPzK3W8CFgIfiHdTFXWpucLYMRTGjsk6jIaT%0AxnUWXwA+Z2ariY5h3JdCGyInVdzcRXFzV9ZhNJxEruB092eBZ+PlNcAlSXxfEakfuoJT+rzCxAlZ%0Ah9AUlCykzyuu25B1CE1ByUJEgihZiEgQJQsRCaJkISJBlCwyUhjflnUIDSU3bQq5aVOyDqOh6UlZ%0AGSlu7Mw6hIbSvXRF1iE0PI0sRCSIkoU0pdygQeQGDco6jD5F0xBpSt379mUdQp+jkYWIBFGyEJEg%0AShYZyo8+g/zoM7IOQySIkoWIBFGyqKHCWeOPel3aspXSlq0ZRSPSO0oWVcoNHBi8b3H9xhQjEUmX%0AkoWIBFGyqFL3/v1ZhyApy7W2HlX+sFnpoixpWNa/P36wrCZNLh997i4FfLGBRf9Ly0sfNjONLEQk%0ASLW1Toea2WNmtsLMXjGzy8xsuJk9bWar4s/DkgpWmk9+1KiKv/aoUQVEI4pTjSri0oe4h+3fRKod%0AWdwN/MLdpwDnE1VTvx1Y4O6TgQXxa5GKlLZtq1lb+RHDKYwZTWHM6JPuY4UCVmjO2Xs1tU6HAO8i%0ALiLk7ofcfTcwm6h6OqiKukjDqCZFtgPbgO+b2fnAYuBWYLS7b4736QJOnqabXO6CqUBczFcyV9qx%0A85T7eLFYg0jqUzXJogBMB/7e3TvM7G6OmXK4u5vZCYsem9lcYC5AK+EXNvVFPcPaYteWo9YrSdS3%0A/IjhAFi/fsDxv79mU80xi06g09074tePESWPLWY2FiD+fMLrmd39Xnef4e4z+tG/ijBEpBaqqaLe%0ABWw0s3PiVVcCy4H5RNXTQVXUgeg/UrP/VwphF52LXXRuRV9baBtHoW1c1THkJ7WTn9TOoatmUNqx%0Ak9KOnVX9/noOiO6ac1nVsWXN3E84Swj7YrMLgO8BLcAa4GNECehRYAKwHrjB3d9yMjjYhvuldmXF%0AcYgkpeden0SuzO05BVvHOnwBe32nhexb1Tkgd38BmHGCTfrLF2kwuoJTpEz3/v3J3e9zzKiiMO5M%0ACuPOZO+HZybz/WusOa8uEakB69dCbkj0BPHS9h0UN70GwOAfvpZlWBXTyEJEgmhkIZISP3yI0vYd%0AQHTNRshFX/VMIwuRGujriQKULEQkkKYhIgnreapW94EDRy4U8wMH8NffiJbPfRu516Pb57vXbMAP%0AH8om0F5SspC65rPO59CQFgD6//yPGUcTpvzJWsXOTcfvsHgZffEpGZqGiEgQjSykruUOFun/8xez%0ADkPQyEJSVLpiOqUrpgfvn598NvnJZ7Pz42/edOWLl6URmlRAyUJEgmgaIqnJL1zSq/1Lq9YAMDz+%0ALPVFIwsRCaJkISJBlCxEJIiShYgEUbIQkSBKFinqeTKSSCNQshCRIEoWKdl6yyyKm1478ii1t9JT%0AzEaknilZpOSMb/8ueN/yB6PkBw9OIxyRqlWVLMzss2a2zMxeNrOHzazVzNrNrMPMVpvZI2bWklSw%0AIpKdaqqojwM+Dcxw93cAeeBG4E7gG+4+CdgF3JxEoI3s9Q9eemS5tHdvhpGInFy194YUgAFmdhgY%0ACGwG3gt8ON7+IPBl4LtVttOQVj1wUbSwxxg8ZhYAY+5ZRG7oEABK27ZlFVpdKl0xnZbO3dGy7h+p%0AuWpqnW4Cvg5sIEoSe4DFwG5376lL3wlUX4BSRDJX8cjCzIYBs4F2YDfwY+DqXnz9XGAuQCsDKw2j%0Az1r3lcuYc+GvAXhg0Sw6bo8GX+eP+jRnfen3WYZWFw5dFVXF3HJJCxO/syJauXBJn3wcXaOoZhry%0APmCtu28DMLPHgcuBoWZWiEcXbcAJHkII7n4vcC9EhZGriKNPGrHUWXXFGQCMHL2Xa+Z8EoCznlGi%0AAGh5ahEA459CCaJOVHM2ZAMw08wGmpkRFUNeDiwEPhDvMwf4WXUhikg9qHhk4e4dZvYYsAQoAs8T%0AjRT+F/iRmf1TvO6+JAJtNLum5JgwILq+YtN/TKbfM33jydVZ2PCl6ODvgbbDvP3+6BH6PLc0w4ia%0Ak7lnPwMYbMP9Ursy6zBqqjC+jR3vbgNg2ONLj1Tutn4tvHrXhQAMeC1P27+EX9zViNb+82V8/YMP%0AAnD9afu56swLMo6osXT4Avb6TgvZV1dwikgQPYMzI8WNnWy9OBpZHPzgBF5/I6piNXjQfgb8Lg/Q%0A3KOKXD7+DP/+ib8G4Nu/fj7DgETJIkNjfxtNAbtGDMB3RVfF5385gLYHmjhJ9OiOzoEMWw65siRh%0A/aKf0+E/O4/CrxZnElqz0jRERILoAGcdyLW2HrnEu9i1JeNo6ov1ayF39gQADowfwr7P7ANg3/7+%0AnHXDS1mG1hB6c4BT05A60H3gAN1dB069YxPyw4f4v/ZhAOydUKD4zEgAiuOy/yfXbDQNEZEgShZS%0A9was3cWAtbsoDjBwwOH0DUZu0CBygwZlHV7T0DRE6l5p5WoAxm3eik+IHoBsG16jtG9fr75PvufW%0A/917kg2wSWhkISJBNLKQPqO0dy+8XNmTxDrnzWrui9wSoGTRh/QMo621VadYA736nUsAePunlCiq%0ApWmIiATRyKLO5EeO4NA7zoqWn11y1LY3D8zpAF2Irs/O0ogiQUoWdaa0fQf5Z3dkHUZqcq2tdB9I%0A7wK0V++5hEmTNwMw5koliiRpGiIiQTSykJpKY1Sx+XOzeL09ukv17Z/sSPz7S0TJQuperjV61kdu%0A2FBK46J7Q/Lb91JctwGAsXdpulELmoaISBCNLCR1+WHDKO3addz60IOdPft0b+6CzV1A9IRoqS0l%0AC0nNkXsxjkkUPU+7SvOsiCTvlNMQM7vfzLaa2ctl64ab2dNmtir+PCxeb2b2rbiC+lIzm55m8CJS%0AOyHHLB7g+LKEtwML3H0ysCB+DXANMDn+mIsKIje17vY2utvbjlufP3M0+TNHZxCRVOOU0xB3/42Z%0ATTxm9WzgPfHyg8CzwBfi9T/w6Fl9z5nZUDMb6+6bkwpY+g5/ftlx63KDBlFcvzGDaKRalZ4NGV2W%0AALqAnn8T44Dyd4KqqIs0iKpPncajiF4/ENHM5prZIjNbdJiD1YYhdSY/ePAJ13f38oE1Uj8qPRuy%0ApWd6YWZjga3x+k3A+LL9VEW9SfnZbRSHDwBQfY8GUenIYj5RhXQ4ulL6fOAj8VmRmcAeHa8QaQyn%0AHFmY2cNEBzNHmlkn8A/AV4FHzexmYD1wQ7z7k8C1wGpgP/CxFGKWPqD7heW6iKfBhJwN+dBJNh1X%0AFSg+fnFLtUGJSP3RvSEiEkTJQkSCKFmISBAlCxEJomQhIkGULEQkiJKFiARRshCRIEoWIhJEyUJE%0AgihZiEgQJQsRCaJkISJBlCxEJIiShYgEUbIQkSBKFiISRMlCRIIoWdRAfuiQI3U/RfoqPVO1Bkq7%0A92QdgkjVNLIQkSCVVlH/mpmtiCul/9TMhpZtmxdXUV9pZlelFbiI1FalVdSfBt7h7tOAV4F5AGY2%0AFbgRODf+mu+YWT6xaCVYYeKEoP1y06aQmzYl5WikEZwyWbj7b4Cdx6z7pbsX45fPEZUphKiK+o/c%0A/aC7ryUqNnRJgvGKSEaSOMD5ceCReHkcUfLooSrqGSmu2xC0X/fSFUeWc6edBsD2G6aRPxyVn23d%0AWeK05VuifQcPPLL/5ttmMWRNCYCBP+1ILG6pX1UlCzO7AygCD1XwtXOBuQCtDKwmDKmCFaK3QOny%0A8/DD3QAMWXeQlq6o2vmhMYM4NH5EtDy0HwNeaQFg7L/9LoNoJUsVJwsz+yhwHXBlXLYQVEVdpGFV%0AlCzM7Grg88C73X1/2ab5wA/N7C7gTGAy8Ieqo5TUeDE69FTYcxDvFx2LLrzaiQ0YAEDLC9sp7doF%0AQCugrN68Kq2iPg/oDzxtZgDPufsn3X2ZmT0KLCeantzi7qW0gpfqFdrPAsAPHqb7heUA6BcmJ1Jp%0AFfX73mL/rwBfqSYoEak/uty7yRXXrj/h+vy55wBQWrayluFIHVOykOMUJk6gOKh/1mFIndG9ISIS%0ARCMLOWLPTTMByBVh0CPPnWJvaTZKFgJAbuBAhv9hKwClVWsyjkbqkaYhIhJEIwsBoHv/ftCIQt6C%0ARhYiEkTJQkSCKFmISBAlCxEJomQhIkGULEQkiJKFiARRshCRIEoWIhJEyUJEgihZiEgQJQsRCaJk%0AISJBlCxEJEhFVdTLtt1mZm5mI+PXZmbfiquoLzWz6WkELSK1V2kVdcxsPPB+oLyo5jVEhYUmE5Um%0A/G71IYpIPaioinrsG0RVycqLVM0GfuCR54ChZjY2kUhFJFMVHbMws9nAJnd/8ZhN44CNZa9VRV2k%0AQfT6sXpmNhD4ItEUpGKqoi7St1Qysngb0A68aGbriCqlLzGzMfSyirq7z3D3Gf1QQRuRetfrZOHu%0AL7n7Ge4+0d0nEk01prt7F1EV9Y/EZ0VmAnvcfXOyIYtIFkJOnT4M/B44x8w6zezmt9j9SWANsBr4%0AL+BTiUQpIpmrtIp6+faJZcsO3FJ9WCJSb3QFp4gEUbIQkSBKFiISRMlCRIIoWYhIECULEQli0dnO%0AjIMw2wa8AWzPoPmRarfh21a7J3eWu48K2bEukgWAmS1y9xlqtzHbzbJttZsMTUNEJIiShYgEqadk%0Aca/abeh2s2xb7Sagbo5ZiEh9q6eRhYjUMSULEQmiZCEiQZQsRCSIkoWIBPl/7hxsB+c/sxIAAAAA%0ASUVORK5CYII=)



This one looks like a "bright green dot" detector, useful to encode cat eyes. At this point, let's go and plot a complete visualization of all the activations in the network. We'll extract and plot every channel in each of our 8 activation maps, and we will stack the results in one big image tensor, with channels stacked side by side.

In [10]:

```
import keras

# These are the names of the layers, so can have them as part of our plot
layer_names = []
for layer in model.layers[:8]:
    layer_names.append(layer.name)

images_per_row = 16

# Now let's display our feature maps
for layer_name, layer_activation in zip(layer_names, activations):
    # This is the number of features in the feature map
    n_features = layer_activation.shape[-1]

    # The feature map has shape (1, size, size, n_features)
    size = layer_activation.shape[1]

    # We will tile the activation channels in this matrix
    n_cols = n_features // images_per_row
    display_grid = np.zeros((size * n_cols, images_per_row * size))

    # We'll tile each filter into this big horizontal grid
    for col in range(n_cols):
        for row in range(images_per_row):
            channel_image = layer_activation[0,
                                             :, :,
                                             col * images_per_row + row]
            # Post-process the feature to make it visually palatable
            channel_image -= channel_image.mean()
            channel_image /= channel_image.std()
            channel_image *= 64
            channel_image += 128
            channel_image = np.clip(channel_image, 0, 255).astype('uint8')
            display_grid[col * size : (col + 1) * size,
                         row * size : (row + 1) * size] = channel_image

    # Display the grid
    scale = 1. / size
    plt.figure(figsize=(scale * display_grid.shape[1],
                        scale * display_grid.shape[0]))
    plt.title(layer_name)
    plt.grid(False)
    plt.imshow(display_grid, aspect='auto', cmap='viridis')
    
plt.show()
```



## **视频识别怎样理解？其实，我们可以将其可视化！**

综述：

本文主要描述的是为视频识别设计的深层网络的显著图（saliency maps）。从早前的论文《卷积神经网络的可视化》（European conference on computer vision. Springer, Cham, 2014）、《可识别定位的深度特征学习》（In CVPR, 2016），以及《Grad-cam:何出此言？基于梯度定位的深度网络视觉解释》（arXiv preprint arXiv:1610.02391 (2016). In ICCV 2017）可以看出，显著图能够有助于可视化模型之所以产生给定预测的原因，发现数据中的假象，并指向一个更好的架构。



#### **什么是可解释性？**

  http://www.sohu.com/a/215753405_465975

我们应该把可解释性看作人类模仿性（human simulatability）。如果人类可以在合适时间内采用输入数据和模型参数，经过每个计算步，作出预测，则该模型具备模仿性（Lipton 2016）。

这是一个严格但权威的定义。以医院生态系统为例：给定一个模仿性模型，医生可以轻松检查模型的每一步是否违背其专业知识，甚至推断数据中的公平性和系统偏差等。这可以帮助从业者利用正向反馈循环改进模型。

#### 树正则化  --斯坦福完全可解释深度神经网络：你需要用决策树搞点事

其论文《Beyond Sparsity: Tree Regularization of Deep Models for Interpretability》已被 AAAI 2018 接收。



很幸运，学界人士也提出了很多对深度学习的理解。以下是几个近期论文示例：

  http://www.sohu.com/a/215753405_465975

- Grad-Cam（Selvaraju et. al. 2017）：使用最后卷积层的梯度生成热力图，突出显示输入图像中的重要像素用于分类。
- LIME（Ribeiro et. al. 2016）：使用稀疏线性模型（可轻松识别重要特征）逼近 DNN 的预测。
- 特征可视化（Olah 2017）：对于带有随机噪声的图像，优化像素来激活训练的 DNN 中的特定神经元，进而可视化神经元学到的内容。
- Loss Landscape（Li et. al. 2017）：可视化 DNN 尝试最小化的非凸损失函数，查看架构／参数如何影响损失情况。
- 

## 参考资料

  

- Deconvolution [Visualizing and Understanding Convolutional Networks](https://arxiv.org/abs/1311.2901)
- Guided-backpropagation [Striving for Simplicity: The All Convolutional Net](https://arxiv.org/abs/1412.6806)
- CAM [Learning Deep Features for Discriminative Localization](https://arxiv.org/abs/1512.04150)
- [Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization](https://arxiv.org/abs/1610.02391)
- [Yes, Deep Networks are great, but are they Trustworthy?](https://ramprs.github.io/2017/01/21/Grad-CAM-Making-Off-the-Shelf-Deep-Models-Transparent-through-Visual-Explanations.html)
- [CAM的tensorflow实现](https://github.com/philipperemy/tensorflow-class-activation-mapping)
- [Grad-CAM的tensorflow实现](https://github.com/insikk/Grad-CAM-tensorflow)









## 参考文献



 ....

 未完待续！

 
