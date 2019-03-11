论文[Timeseries data mining](https://dl.acm.org/citation.cfm?id=2379788)（2012）中提出：时间序列数据挖掘包括7个基本任务和3个基础问题：

||Task|
|---|---|
|1|query by content|
|2|clustering|
|3|classfication|
|4|segmentation|
|5|prediction|
|6|anomaly detection|
|7|motif discovery|

||Issues|
|---|---|
|1| data representation|
|2| similarity measure|
|3| indexing|


现已有2013-2018年间重要会议的时间序列相关论文列表（见下文Paper List）。

接下来需要我们快速阅读每篇论文的Abstract和Introduction，按照“新问题”和“新方法”对论文进行分类。
其中新方法的论文暂时放一边，重点关注新问题，总结记录2013-2018年论文中提出的新问题。

* 新问题关注度 > 新方法关注度
* 提出新问题的论文的工作量<提出新方法的论文的工作量，因为后者需要battle所有已有的方法
* 问题可能和具体应用高度相关，也可能是一般性的问题

最后，了解一下Introduction的典型结构有助于快速阅读，例如：
1. 大量的时间序列产生
2. 在工业时间序列中 工况需要分段
3. 现在是人工来做这件事，也有一些其它自动化方法，但是存在问题缺陷不足
4. 这件事情non-trivial 有难度
5. 我们的方法怎么对应上面的non-trivial 一些结果 在数据集上验证
6. 我们的contributions，可能是提出了一个新问题、提出了一种改进算法等等
7. 后文的结构

# 分工
芮: prediction(47+)

康: anomaly detection(26+), motif discovery(10+), analysis(2+)

江: query by content(16+), classfication(23+)

安: clustering(28+), segmentation(3+)

# 2013-2019新问题（under construction）
## clustering & segmentation
* Similarity Measure Selection for Clustering Time Series Databases：对时间序列数据集进行聚类分析时，自动选择最合适的相似性度量方法；
* Interpretable Categorization of Heterogeneous Time Series Data：多种类多维度时间序列的解释性分类；
* Developing a Low Dimensional Patient Class Profile in Accordance to Their Respiration-Induced Tumor Motion：利用肿瘤数据（由时间序列组成）创建低维的病人档案（包含病人个人信息、生命体征以及化验或检查结果）；
* Local Search Methods for k-Means with Outliers：带有异常值的k-Means的本地搜索；
* Density Based Clustering over Location Based Services：LBS（定位服务）基于密度的聚类；

## query by content
* 任意测度下的AkNN
* 分布式的滑动窗口中近似矩阵计算 (几乎新问题)
* 随机长度直方图化流的子序列匹配
* 对轨迹做逆KNN
* 用弱标记的事件序列来预测（Using Weakly Labeled Time Series to Predict Outcomes）

## classfication
注：时间序列分类是一个定义比较明确的问题，由此衍生出完全新问题的可能性很小。有些所谓的新问题只是在特定领域的运用。
对于没有在标题中概括方法的研究，分类一栏额外给出了对其方法的概括方便进一步分类。 有一部分工作与其说是新方法不如说是老方法在某些特定领域的使用，这些被分类为“新方法？”。
* 对TSC使用AL (几乎新问题)
* 脑电图数据选择
* 用TSC解决goog abandonment

## prediction
* 更准确地刻画数据的动态特性（a more accurate characterization of data dynamics）
    - 捕捉隐藏因素的时变模式（capturing evolution patterns of hidden factors）AR model with IID Gaussian distribution to simulate the white noises that drive the diffusion of target data -> Brownian motion -> OU process & SDE (continuous-time domain) ->mixture of evolving factors over time, no single one that would presistently drive the time series through time
* 数据类型：时空数据（Space-Time series forecasting / Time series exhibiting spatial dependencies / spatial temporal）
    - 工业大规模时空预测问题，兼具准确性和灵活度要求（ industrial large-scale spatio-temporal prediction problems with both accuracy and flexibility requirements）。大规模在线出租车行业（large-scale online taxicab industries），预测UOTD，在精确度要求之外还要求有足够的灵活度，以应对该行业经常性的政策规范或商业战略的变化。（predict the Unit Original Taxi Demand (UOTD), which refers to the number of taxi-calling requirements submitted per unit time (e.g., every hour) and per unit region (e.g., each POI). In the fast-developing online taxicab industry, application and key factor changes due to new regulations or business strategies are common and frequent. ）两个范式：（1）复杂模型+少量features，（2）简单模型+大量features。To accurately predict UOTD while remaining flexible to scenario changes，选择（2）。
* 领域强相关
    - 基于电子病历的诊断预测（diagonosis prediction: to predict the future diagnoses based on patient’s historical EHR data）。两点挑战：电子病历序列数据中的时变和高维特性；预测结果的可解释性（The most important challenges for this task are 1, to model the temporality and high dimensionality of sequential EHR data and 2, to interpret the prediction results.）
    - 众筹任务动态的追踪和预测，而不是只预测众筹是否成功的一个最后结果。（Tracking and forecasting the dynamics in crowdfunding instead of a final result; A special goal is to forecast the funding amount for a given campaign and its perks in the future days）
    - 空中和海上交通移动实体的事件及轨迹的实时检测和预测问题（real-time detection and prediction of events and trajectories over multiple heterogeneous, voluminous, fluctuating, and noisy data streams of moving entities in aerial and maritime transportation）
    - 用线下预测来辅助在线任务分配。Flexible Two-sided Online task Assignment (FTOA), a new problem of online task assignment in real-time spatial data that is fit for practical O2O applications where workers are allowed to move around if no task is assigned
     - 车辆轨迹多步预测（multi-step vehicle trajectory prediction）
        + 意义：有利于location-based services，并且比一步预测更好
        + 难点：当新模式出现或者历史轨迹不完整（when new patterns appear or the previous trajectory is incomplete）
        + 方法：feature engineering，利用道路内(intra-road)和道路间(inter-road)特征(road-aware features)
* 放松限制性假设（relax restrictive assumptions），减少人工干预，提高自动化（perform anomaly detection and forecasting robustly without human intervention / automated algorithm for anomaly detection and/or forecasting）
    - 常见的限制性假设包括数据的周期性已知、无异常（尖峰或水平变化）窗口存在，然后人工地把这些先验知识加到异常监测和预测系统中。放松这些假设的一个方法：sparse decomposition model + ARMA noise model: jointly estimating the latent components (viz. seasonality, level changes, and spikes) in the observed time series without assuming the availability of anomaly-free time windows. 
* 在线或流数据场景（online or streaming setting）
* 预测变量/特征的选择 feature engineering（constructing a set of predictor variables that can be used in a forecast model / multivariate time series forecasting）
* 从一个or一堆时间序列学习
    - clinical time series preditcion: population models vs patient-specific models; the adaptive model switching approach
    - collaborative sequence prediction problem: Taking the correlation between users’ behavior sequences into account, we frame the sequential recommendation as the collaborative sequence prediction problem.

## anomaly detection
* 传统的异常检测都是基于“点”的，如何对于“异常片段”进行度量？本文介绍了一种新的measurement。
* 领域强相关：通过挖掘历史飞行数据来开发决策支持工具，以主动识别和管理飞行中遇到的高风险情况
* （在新的场景下）：在“交互式环境”中解决异常值检测问题的系统
* 在许多应用场景中，希望发现社交媒体数据基于地理和/或时间分区的模式和趋势，进行时空情感分析
* 存在异常值的情况下k均值聚类的问题
* 老问题-新假设：观察时间序列中的潜在成分，放宽了已有方法的假设约束
* 领域相关：电力预测
* 第一个分布式轨迹流上的异常检测
* 对于文本序列数据，引入了一个强大的时间序列链定义，以及一个可扩展的算法
* 在异常检测中引入了时间因果依赖性的考虑，并针对符合该假设的数据集进行实验
* 入侵检测系统中低级异构事件的研究
* 时间序列异常检测中使用迁移学习

## motif discovery
* 新假设：在motif中引入额外的上下文信息
* 如何在大数据场景下进行motif检测
* 以前工作不能同时进行相关性计算（fast correlation computations ）和 剪枝（prune subsequence pairs）
* 对于motif，加上了stream的场景



# Paper List
* Task 1 query by content (16+)
* Task 2 clustering (28+)
* Task 3 classfication (23+)
* Task 4 segmentation (3+)
* Task 5 prediction (47+)
* Task 6 anomaly detection (26+)
* Task 7 motif discovery (10+)
* Task 8 analysis (2+)


## Task 1 query by content (16+)

### representation

### similarity measure

|Source|Title|Classification|
|---|---|---|
|TKDE 2016|Metric All-k-Nearest-Neighbor Search|新问题：任意测度下的AkNN|
|NIPS 2017|Soft-DTW: a Differentiable Loss Function for Time-Series|新方法|
|NIPS 2018|Autowarp: Learning a Warping Distance from Unlabeled Time Series Using Sequence Autoencoders|新方法|

### indexing

### MORE
|Source|Title|Classification|
|---|---|---|
|SIGIR-2018|CA-LSTM: Search Task Identification with Context Attention based LSTM|新方法|
|SIGMOD-2018|Qetch: Time Series Querying with Expressive Sketches|新方法|
|SIGMOD-2017|Approximate Query Processing: No Silver Bullet|survey|
|SIGMOD-2017|Approximate Query Engines: Commercial Challenges and Research Opportunities|Keynote|
|SIGMOD-2017|Approximate Query Processing for Interactive Data Science|Keynote|
|SIGMOD-2015|SMiLer: A Semi-Lazy Time Series Prediction System for Sensors|新方法|
|VLDB-2018|Locality-Sensitive Hashing for Earthquake Detection: A Case Study Scaling Data-Driven Science|新方法|
|VLDB-2017|DITIR: Distributed Index for High Throughput Trajectory Insertion and Real-time Temporal Range Query|新方法|
|DASFAA-2018|Time-Based Trajectory Data Partitioning for Efficient Range Query|新方法|
|ICDE-2017|Tracking Matrix Approximation over Distributed Sliding Windows. 833-844|几乎新问题（1）：分布式的滑动窗口中近似矩阵计算|
|ICDE-2015|Predictive tree: An efficient index for predictive queries on road networks. 1215-1226|新方法|
|TKDE 2017|Measuring Concentration of Distances—An Effective and Efficient Empirical Index.|新方法|
|TKDE 2018|Non-Overlapping Subsequence Matching of Stream Synopses|新问题:随机长度直方图化流的子序列匹配|
|TKDE 2018|Reverse k Nearest Neighbor Search over Trajectories|新问题:对轨迹做逆KNN|
|TKDE 2018|Fast Cosine Similarity Search in Binary Space with Angular Multi-Index Hashing|新方法|


## Task 2 clustering (28+)

### survey

|Source|Title|Classification|
|---|---|---|
|TODS 2013|Data Stream Clustering: A Survey|新方法|

### representation

### similarity measure

|Source|Title|Classification|
|---|---|---|
|TKDE 2016|Similarity Measure Selection for Clustering Time Series Databases|新问题|
|TKDE 2018|Similarity Metrics for SQL Query Clustering|新方法|
|TODS 2017|Fast and Accurate Time-Series Clustering|新方法|

### indexing
### MORE

|Source|Title|Classification|
|---|---|---|
|DASFAA-2018|Scalable Active Constrained Clustering for Temporal Data|新方法|
|ECML PKDD-2017|Identifying Representative Load Time Series for Load Flow Calculations|新方法|
|ICDM-2017|Distance and Density Clustering for Time Series Data|新方法|
|PODS-2018|Subtrajectory Clustering: Models and Algorithms|新方法|
|SDM-2018|Interpretable Categorization of Heterogeneous Time Series Data|新问题|
|SIGIR-2018|CA-LSTM: Search Task Identification with Context Attention based LSTM|新方法|
|SIGKDD-2017|(Research Track最佳论文Runner Up)Toeplitz Inverse Covariance-Based Clustering of Multivariate Time Series Data|新方法|
|SIGKDD-2017|Effective and Real-time In-App Activity Analysis in Encrypted Internet Traffic Streams|新方法|
|SIGKDD-2017|Patient Subtyping via Time-Aware LSTM Networks|新方法|
|SIGKDD-2017|Robust Spectral Clustering for Noisy Data|新方法|
|SIGKDD-2017|Clustering Individual Transactional Data for Masses of Users|新方法|
|SIGKDD-2017|KATE: K-Competitive Autoencoder for Text|新方法|
|VLDB-2018|Clustering Stream Data by Exploring the Evolution of Density Mountain|新方法|
|VLDB-2017|Developing a Low Dimensional Patient Class Profile in Accordance to Their Respiration-Induced Tumor Motion|新问题|
|VLDB-2017|NG-DBSCAN: Scalable Density-Based Clustering for Arbitrary Data. 157-168|新方法|
|VLDB-2017|Local Search Methods for k-Means with Outliers. 757-768|新问题|
|VLDB-2017|Dimensions Based Data Clustering and Zone Maps. 1622-1633|新方法|
|VLDB-2015|YADING: Fast Clustering of Large-Scale Time Series Data. 473-484 ★★★|新方法|
|ICDE-2016|Streaming spectral clustering. 637-648|新方法|
|ICDE-2017|Density Based Clustering over Location Based Services. 461-469|新问题|
|ICDE-2017|A model-based approach for text clustering with outlier detection. 625-636|新方法|
|ICDE-2017|Accelerating large scale centroid-based clustering with locality sensitive hashing. 649-660|新方法|
|ICDE-2017|PurTreeClust: A purchase tree clustering algorithm for large-scale customer transaction data. 661-672|新方法|
|ICDE-2017|ClEveR: Clustering events with high density of true-to-false occurrence ratio. 918-929|新方法|

## Task 3 classfication (23+)
注：时间序列分类是一个定义比较明确的问题，由此衍生出完全新问题的可能性很小。有些所谓的新问题只是在特定领域的运用。  
对于没有在标题中概括方法的研究，分类一栏额外给出了对其方法的概括方便进一步分类。
有一部分工作与其说是新方法不如说是老方法在某些特定领域的使用，这些被分类为“新方法？”。
### representation

|Source|Title|Classification|
|---|---|---|
|TKDE 2016|Classifying Time Series Using Local Descriptors with Hybrid Sampling|新方法|
|TKDE 2015|Time-Series Classification with COTE: The Collective of Transformation-Based Ensembles.|新方法|
|TKDE 2014|Probabilistic Sequence Translation-Alignment Model for Time-Series Classification|新方法|

### similarity measure

### indexing

### MORE

|Source|Title|Classification|
|---|---|---|
|SIGKDD-2017|Effective and Real-time In-App Activity Analysis in Encrypted Internet Traffic Streams|新方法:分窗+抽特征+聚类+随机森林|
|ICDE-2017|ACTS: An Active Learning Method for Time Series Classification|几乎新问题（2）：对TSC使用AL|
|ICDE-2017|Time Series Classification by Sequence Learning in All-Subsequence Space|新方法：在子序列空间中使用梯度下降找到最具区分力的子序列|
|VLDB-2017|Effects of Varying Sampling Frequency on the Analysis of Continuous ECG Data Streams|新方法|
|VLDB-2017|Matrix Profile IV: Using Weakly Lab1eled Time Series to Predict Outcomes|新问题|can we learn from the weakly labeled time series|
|SDM-2018|Interpretable Categorization of Heterogeneous Time Series Data|新方法：使用扩展的决策树|
|SDM-2018|Evolving Separating References for Time Series Classification|新方法|
|SDM-2018|Classifying Multivariate Time Series by Learning Sequence-level Discriminative Patterns|新方法|
|SDM-2018|Brain EEG Time Series Selection: A Novel Graph-Based Approach for Classification|新问题？：脑电图数据选择|
|ICDM-2017|Linear Time Complexity Time Series Classification with Bag-of-Pattern-Features|新方法(注：优于DTW）|
|EDBT-2018|Extracting Statistical Graph Features for Accurate and Efficient Time Series Classification|新方法：时间序列转为图，在图上抽特征|
|DASFAA-2018|Nearest Subspace with Discriminative Regularization for Time Series Classification|新方法：降维+近邻子空间分类+模型编译|
|CIKM-2017|Fast and Accurate Time Series Classification with WEASEL|滑动窗口抽特征+特征选择+逻辑回归|
|CIKM-2017|Does That Mean You're Happy?: RNN-based Modeling of User Interaction Sequences to Detect Good Abandonment|新问题：用TSC解决goog abandonment|
|ECML PKDD-2017|Behavioral Constraint Template-Based Sequence Classification|新方法|
|ECML PKDD-2017|Cost Sensitive Time-Series Classification|新方法：shaplet+学习FP，FN各自的权重|
|ECML PKDD-2017|Efficient Temporal Kernels Between Feature Sets for Time Series Classification|新方法|
|ECML PKDD-2017|Analyzing Granger Causality in Climate Data with Time Series Classification Methods|新方法？|
|ECML PKDD-2017|End-to-end Learning of Deep Spatio-temporal Representations for Satellite Image Time Series Classification|新方法？|
|ECML PKDD-2017|Temporal and spatial approaches for land cover classification|新方法？|
|ECML PKDD-2017|Self-Adaptive Ensemble Classifier for Handling Complex Concept Drift|新方法|


## Task 4 segmentation (3+)

### representation


|Source|Title|Classification|
|---|---|---|
|TKDE 2017|Efficient Pattern-Based Aggregation on Sequence Data.|新方法|
|TKDE 2014|An Adaptive Approach to Real-Time Aggregate Monitoring With Differential Privacy|新方法|

### similarity measure

### indexing

|Source|Title|Classification|
|---|---|---|
|TKDE 2018|BEATS: Blocks of Eigenvalues Algorithm for Time Series Segmentation.|新方法|



## Task 5 prediction (47+)

### representation

|Source|Title|Classification|
|---|---|---|
|NIPS 2018|Deep State Space Models for Time Series Forecasting|新方法|a novel approach to probabilistic time series forecasting that combines state space models with deep learning|

### similarity measure

### indexing

### MORE

|Source|Title|Classification|Notes|
|---|---|---|---|
|SIGKDD-2017|Incremental Dual-memory LSTM in Land Cover Prediction|新方法|技术上是时间序列分类，用分类来做预测，预测结果是categorical variable。和传统分类问题相比，land cover prediction的三点数据异质性挑战：（1）空间局部性，（2）时变性，（3）突变不可知zero-shot|
|SIGKDD-2017|Mixture Factorized Ornstein-Uhlenbeck Processes for Time-Series Forecasting|新问题|stock prices & sensor streams；AR model with IID Gaussian distribution to simulate the white noises that drive the diffusion of target data -> Brownian motion -> OU process & SDE (continuous-time domain) ->mixture of evolving factors over time, no single one that would presistently drive the time series through time；stock prices & sensor streams；related提到三种AR变种方法直接处理非平稳时间序列|
|SIGKDD-2017|Retrospective Higher-Order Markov Processes for User Trails|新方法|user trails；一阶和高阶MC，bias&variance，一阶不如高阶，而高阶有三点问题：（1）参数数量与阶次呈指数比，（2）参数多相应需要的训练数据也指数增加，（3）历史数据长度参数m不好确定，m定得过大容易过拟合，模型复杂泛化能力差不可信；a simplied, special case of a higher-order Markov chain，a low-parameter model；预测准确度对标higher-order Markov chains & Kneser-Ney regularization & tensor factorizations；训练过程可以并行|
|SIGKDD-2017|The Simpler The Better: A Unified Approach to Predicting Original Taxi Demands on Large-Scale Online Platforms|新问题，新方法|https://www.youtube.com/watch?v=OlZhSrdU3IA ；To accurately predict UOTD while remaining flexible to scenario changes；两个paradigm: 1，复杂模型+少量features，2，简单模型+大量features。在出租车业务场景中要素经常变化，因此选择后者，transform model redesign to feature redesign，故而这篇文章的难点也就集中在feature engineering；简单的线性回归模型+两千万features+parallel and scalable的optimization technique；强调自己是一个pilot study，可以为其它类似的大规模时空预测兼准确度和灵活度需求的问题提供insights|
|SIGKDD-2017|Stock Price Prediction via Discovering Multi-Frequency Trading Patterns|新方法|a novel State Frequency Memory (SFM) recurrent network to capture the multi-frequency trading patterns from past market data to make long and short term predictions over time；受DFT和LSTM的启发；第一个这样做的，novel|
|SIGKDD-2017|Dipole: Diagnosis Prediction in Healthcare via Attention-based Bidirectional Recurrent Neural Networks|新问题，新方法|基于电子病历的诊断预测（diagonosis prediction: to predict the future diagnoses based on patient’s historical EHR data）。两点挑战：（1）电子病历序列数据中的时变和高维特性，（2)预测结果的可解释性（The most important challenges for this task are 1, to model the temporality and high dimensionality of sequential EHR data and 2, to interpret the prediction results.）|
|SIGKDD-2017|Tracking the Dynamics in Crowdfunding|新问题|hierarchical time series: campaign-level dynamics and perk-level dynamics；用switching regression来解决异质性heterogeneity；感觉ad-hoc|
|SIGKDD-2017|DeepMood: Modeling Mobile Phone Typing Dynamics for Mood Detection|新方法|验证了用移动手机对话情景下的按键动态数据来评估预测情绪障碍、严重程度的方法的可行性|
|SDM-2018|Sparse Decomposition for Time Series Forecasting and Anomaly Detection|新问题，新方法|sparse decomposition and ARMA noise model优势互补；latent components包括trends, spikes, seasonalities；好处包括（1）先时间序列分解，有利于后续的异常监测和预测，（2)放宽限制性条件从而有助于算法自动化而不用人工干预；实验对标的方法包括ETS&ARIMA|
|SDM-2018|StreamCast: Fast and Online Mining of Power Grid Time Sequences|新方法|建模和预测; physics-based model有利于推理; online or streaming setting|
|SDM-2018|Who will Attend This Event Together? Event Attendance Prediction via Deep LSTM Networks|新方法|feature engineering，人工分析抽取了三种特征：semantic, temporal, spatial|
|SIGMOD-2018|Spatiotemporal Traffic Volume Estimation Model Based on GPS Samples|新方法|real time volumn estimation/traffic monitoring；两种数据：（1）静态的专门安装的传感器，（2）Probe Vehicle Data (PVD) "水流中的软木塞”；这两种数据可以看成是accuracy和flexibility的trade off：（1）能准确的测量traffic flow，但是只能覆盖固定的地点，并且传感器unreliable，数据缺失，（2）测的是vehicle flow，不能得到准确的交通流，但是它有足够的覆盖度/ubiquitous；本文重点是把两种数据结合起来实现bridge the gap。|
|SIGMOD-2015|SMiLer: A Semi-Lazy Time Series Prediction System for Sensors|新方法|a new method to apply the GP for sensor time series prediction；不在全部数据上训练高斯过程，而是结合query对应的一小部分数据来构造|
|SIGIR-2018|A Flexible Forecasting Framework for Hiera1rchical Time Series with Seasonal Patterns: A Case Study of Web Traffic|新方法|a new flexible framework for hierarchical time series (HTS) forecasting|
|SIGIR-2018|Modeling Long- and Short-Term Temporal Patterns with Deep Neural Networks|新方法|机器学习，多变量时间序列预测问题；CNN+RNN；related提到用AR模型来解决神经网络模型的scale insensitive问题|
|SIGIR-2018|Ad Click Prediction in Sequence with Long Short-Term Memory Networks: an Externality-aware Model|新方法|第一个用LSTM来预测点击率；an Externality-aware Model: considers user browsing behavior and the impact of top ads quality to the current one|
|VLDB-2018|Forecasting Big Time Series: Old and New|tutorial|good||
|VLDB-2017|Flexible Online Task Assignment in Real-Time Spatial Data|新问题|不是时间序列预测，只是有点相关；Flexible Two-sided Online task Assignment (FTOA), a new problem of online task assignment in real-time spatial data that is fit for practical O2O applications where workers are allowed to move around if no task is assigned|
|VLDB-2017|A Time Machine for Information: Looking Back to Look Forward|新问题|不是时间序列预测；goal of building a time machine for information that will record and preserve history accurately, and to help people “look back” and so as to “look forward”|
|ICDT-2018|Short-Term Traffic Forecasting: A Dynamic ST-KNN Model Considering Spatial Heterogeneity and Temporal Non-Stationarity|新方法|accurate and robust short-term traffic forecasting|
|ICDM-2017|Spatio-Temporal Neural Networks for Space-Time Series Forecasting and Relations Discovery|新方法|forecasting time series of spatial processes, i.e. series of observations sharing temporal and spatial dependencies；时空时间序列经典领域：epidemiology, geo-spatial statistics and car-traffic prediction|
|ICDM-2017|Time-Aware Latent Hierarchical Model for Predicting House Prices|新方法||
|ICDM-2017|Autoregressive Tensor Factorization for Spatio-temporal Predictions|新方法|侧重空间预测predict unknown locations|
|ICDM-2017|Deep and Confident Prediction for Time Series at Uber|新方法|Accurate time series forecasting and reliable estimation of the prediction uncertainty|
|ICDM-2017|Improving Multivariate Time Series Forecasting with Random Walks with Restarts on Causality Graphs|新方法|feature engineering, feature selection methods; constructing a set of predictor variables that can be used in a forecast model is one of the greatest challenges in forecasting|
|EDBT-2018|Big Data Analytics for Time Critical Maritime and Aerial Mobility Forecasting|新问题|userdefined challenges in the air-traffic management and maritime domains|
|DASFAA-2018|A Road-Aware Neural Network for Multi-step Vehicle Trajectory Prediction|新方法|multi-step vehicle trajectory prediction task; feature engineering: RA-LSTM: a neural network model (LSTM) combining road-aware features|
|CIKM-2017|Coupled Sparse Matrix Factorization for Response Time Prediction in Logistics Services|新方法|不算时间序列预测，是预测，是物流业务相关的response time predition|
|CIKM-2017|A Personalized Predictive Framework for Multivariate Clinical Time Series via Adaptive Model Selection|新问题，新方法|the adaptive model switching approach；population-based & individual-specific model|
|CIKM-2017|A Study of Feature Construction for Text-based Forecasting of Time Series Variables|新方法|feature engineering, 设计新的feature用于预测任务；股票市场预测&新闻中提取的话题特征|
|CIKM-2017|Collaborative Sequence Prediction for Sequential Recommender|新方法|collaborative sequence prediction problem; the correlation between users’ behavior sequences instead of independence assumption|
|ECML PKDD-2017|BeatLex: Summarizing and Forecasting Time Series with Patterns|新方法||
|ECML PKDD-2017|Arbitrated Ensemble for Time Series Forecasting|新方法||
|ECML PKDD-2017|Forecasting and Granger Modelling with Non-linear Dynamical Dependencies|新方法||
|ECML PKDD-2017|PowerCast: Mining and Forecasting Power Grid Sequences|新方法||
|ECML PKDD-2017|Modeling the Temporal Nature of Human Behavior for Demographics Prediction|新方法||
|ECML PKDD-2017|Taking It for a Test Drive: A Hybrid Spatio-Temporal Model for Wildlife Poaching Prediction Evaluated Through a Controlled Field Test|新方法||
|ECML PKDD-2017|Predicting Defective Engines using Convolutional Neural Networks on Temporal Vibration Signals|新方法||
|ECML PKDD-2017|Usefulness of Unsupervised Ensemble Learning Methods for Time Series Forecasting of Aggregated or Clustered Load|新方法||
|ICDE-2017|Prediction-Based Task Assignment in Spatial Crowdsourcing. 997-1008|新方法||
|ICDE-2017|Discovering interpretable geo-social communities for user behavior prediction. 942-953|新方法||
|ICDE-2016|Link prediction in graph streams. 553-564|新方法||
|ICDE-2015|Searchlight: Context-aware predictive Continuous Querying of moving objects in symbolic space. 687-698|新方法||
|ICDE-2015|Predictive tree: An efficient index for predictive queries on road networks. 1215-1226|新方法||

## Task 6 anomaly detection (26+)

### survey

|Source|Title|Classification|
|---|---|---|
|TKDE 2014|Outlier Detection for Temporal Data: A Survey|survey|

### representation

### similarity measure

|Source|Title|Classification|
|---|---|---|
|NIPS 2018|Precision and Recall for Time Series|新问题：传统的异常检测都是基于“点”的，如何对于“异常片段”进行度量？本文介绍了一种新的measurement。|

### indexing

### MORE

|Source|Title|Classification|
|---|---|---|
|SIGKDD-2017|Anomaly Detection in Streams with Extreme Value Theory|新方法：在这里，我们提出了一种基于极值理论检测流单变量时间序列中的异常值的新方法|
|SIGKDD-2017|Let's See Your Digits: Anomalous-State Detection using Benford's Law|新方法：发现异常数据出现前数值型数据的一些规律|
|SIGKDD-2017|Finding Precursors to Anomalous Drop in Airspeed During a Flight's Take-off|新问题、领域强相关：通过挖掘历史飞行数据来开发决策支持工具，以主动识别和管理飞行中遇到的高风险情况|
|SIGKDD-2017|Distributed Local Outlier Detection in Big Data|新方法：局部离群因子方法（Local Outlier Factor (LOF) method）|
|SIGKDD-2017|REMIX: Automated Exploration for Interactive Outlier Detection|新问题（在新的场景下）：在“交互式环境”中解决异常值检测问题的系统。|
|SIGKDD-2017|Scalable Top-n Local Outlier Detection|新方法：第一个可扩展的Top-N本地离群值检测方法|
|SIGKDD-2017|Compass: Spatio Temporal Sentiment Analysis of US Election|新问题：在许多应用场景中，希望发现社交媒体数据基于地理和/或时间分区的模式和趋势，进行时空情感分析。|
|SIGMOD-2018|TcpRT: Instrument and Diagnostic Analysis System for Service Quality of Cloud Databases at Massive Scale in Real-time|新系统|
|SIGMOD-2018|Auto-Detect: Data-Driven Error Detection in Tables|新方法：检测SQL中的错误|
|SIGMOD-2018|RushMon: Real-time Isolation Anomalies Monitoring|新方法：系统隔离机制不一致性的检测|
|ICDE-2018|Improving Quality of Observational Streaming Medical Data by Using Long Short-Term Memory Networks (LSTMs)|新方法：将LSTM和编解码器用在医疗领域的异常检测|
|ICDE-2017|LSHiForest: A Generic Framework for Fast Tree Isolation Based Ensemble Anomaly Analysis. 983-994|新方法|
|VLDB-2017|Time Series Data Cleaning: From Anomaly Detection to Anomaly Repairing|新方法：宋韶旭老师工作|
|VLDB-2017|Local Search Methods for k-Means with Outliers|新问题：存在异常值的情况下k均值聚类的问题。|
|VLDB-2016|Streaming Anomaly Detection Using Randomized Matrix Sketching. 192-203|新方法|
|SDM-2018|Sparse Decomposition for Time Series Forecasting and Anomaly Detection|老问题-新假设：观察时间序列中的潜在成分，放宽了已有方法的假设约束|
|SDM-2018|StreamCast: Fast and Online Mining of Power Grid Time Sequences|新问题-领域相关：电力预测|
|SDM-2018|Outlier Detection over Distributed Trajectory Streams|新问题：第一个分布式轨迹流上的异常检测|
|ICDM-2017|Matrix Profile VII: Time Series Chains: A New Primitive for Time Series Data Mining (Best Student Paper Award)|新问题：对于文本序列数据，引入了一个强大的时间序列链定义，以及一个可扩展的算法|
|ICDM-2017|Dependency Anomaly Detection for Heterogeneous Time Series: A Granger-Lasso Approach|新问题：在异常检测中引入了时间因果依赖性的考虑，并针对符合该假设的数据集进行实验|
|ICDM-2017|Deep and Confident Prediction for Time Series at Uber|新方法|
|CIKM-2017|Efficient Discovery of Abnormal Event Sequences in Enterprise Security Systems|新问题：入侵检测系统中低级异构事件的研究|
|CIKM-2017|Anomaly Detection in Dynamic Networks using Multi-view Time-Series Hypersphere Learning|新方法|
|ECML PKDD-2017|UAPD: Predicting Urban Anomalies from Spatial-Temporal Data|新框架：城市异常预测（UAPD）框架|
|ECML PKDD-2017|Transfer Learning for Time Series Anomaly Detection|新问题：时间序列异常检测中使用**迁移学习**|

## Task 7 motif discovery (10+)

### representation

### similarity measure

### indexing

### MORE
|Source|Title|Classification|
|---|---|---|
|SIGKDD-2017|Matrix Profile V: A Generic Technique to  Incorporate Domain Knowledge into Motif Discovery |新方法|
|SIGKDD-2017|Contextual Motifs: Increasing the Utility of Motifs using Contextual Data|新假设：在motif中引入额外的上下文信息|
|SIGMOD-2018|Matrix Profile X: VALMOD - Scalable Discovery of Variable-Length Motifs in Data Series|新方法：检测可变长度的motif|
|SIGMOD-2018|VALMOD: A Suite for Easy and Exact Detection of Variable Length Motifs in Data Series|上文的demo|
|ICDM-2017|IterativE Grammar-Based Framework for Discovering Variable-Length Time Series Motifs|新方法|
|ICDM-2017|Efficient discovery of time series motifs with large length range in million scale time series|新方法：检测可变长度的motif|
|ICDM-2017|Matrix Profile VI: Meaningful Multidimensional Motif Discovery|新方法|
|ICDE-2016|Fast motif discovery in short sequences. 1158-1169|新问题：如何在大数据场景下进行motif检测|
|ICDE-2015|Quick-motif: An efficient and scalable framework for exact motif discovery. 579-590|新问题：以前工作不能同时进行相关性计算（fast correlation computations ）和 剪枝（prune subsequence pairs）|
|VLDB-2015|Rare Time Series Motif Discovery from Unbounded Streams. 149-160|新问题：对于motif，加上了stream的场景|

## Task 8 analysis (2+)

### representation

### similarity measure

|Source|Title|Classification|
|---|---|---|
|TKDE 2018|Diverse Relevance Feedback for Time Series with Autoencoder Based Summarizations|新方法|

### indexing

|Source|Title|Classification|
|---|---|---|
|TKDE 2017|Time Series Management Systems: A Survey.|survey：对时序系统的总结|
