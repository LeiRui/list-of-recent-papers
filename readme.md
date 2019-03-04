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
其中新方法的论文暂时放一边，重点关注新问题，总结记录2013-2018年间产生的新问题。

* 新问题关注度 > 新方法关注度
* 提出新问题的论文的工作量<提出新方法的论文的工作量，因为后者需要battle所有已有的方法

最后，了解一下Introduction的典型结构有助于快速阅读，例如：
1. 大量的时间序列产生
2. 在工业时间序列中 工况需要分段
3. 现在是人工来做这件事，也有一些其它自动化方法，但是存在问题缺陷不足
4. 这件事情non-trivial 有难度
5. 我们的方法怎么对应上面的non-trivial 一些结果 在数据集上验证
6. 我们的contributions，可能是提出了一个新问题、提出了一种改进算法等等
7. 后文的结构

# 分工
芮：prediction(47+)

康: anomaly detection(26+), motif discovery(10+), analysis(2+)

江: query by content(16+), classfication(23+)

安: clustering(28+), segmentation(3+)

# 2013-2019新问题（under construction）
* 

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
|TKDE 2016|Metric All-k-Nearest-Neighbor Search||
|NIPS 2017|Soft-DTW: a Differentiable Loss Function for Time-Series||
|NIPS 2018|Autowarp: Learning a Warping Distance from Unlabeled Time Series Using Sequence Autoencoders||

### indexing

### MORE
|Source|Title|Classification|
|---|---|---|
|SIGIR-2018|CA-LSTM: Search Task Identification with Context Attention based LSTM||
|SIGMOD-2018|Qetch: Time Series Querying with Expressive Sketches||
|SIGMOD-2017|Approximate Query Processing: No Silver Bullet||
|SIGMOD-2017|Approximate Query Engines: Commercial Challenges and Research Opportunities||
|SIGMOD-2017|Approximate Query Processing for Interactive Data Science||
|VLDB-2017|DITIR: Distributed Index for High Throughput Trajectory Insertion and Real-time Temporal Range Query||
|DASFAA-2018|Time-Based Trajectory Data Partitioning for Efficient Range Query||
|ICDE-2017|Tracking Matrix Approximation over Distributed Sliding Windows. 833-844||
|ICDE-2015|Predictive tree: An efficient index for predictive queries on road networks. 1215-1226||
|TKDE 2017|Measuring Concentration of Distances—An Effective and Efficient Empirical Index.||
|TKDE 2018|Non-Overlapping Subsequence Matching of Stream Synopses||
|TKDE 2018|Reverse k Nearest Neighbor Search over Trajectories||
|TKDE 2018|Fast Cosine Similarity Search in Binary Space with Angular Multi-Index Hashing||

## Task 2 clustering (28+)

### survey

|Source|Title|Classification|
|---|---|---|
|TODS 2013|Data Stream Clustering: A Survey|

### representation

### similarity measure

|Source|Title|Classification|
|---|---|---|
|TKDE 2016|Similarity Measure Selection for Clustering Time Series Databases||
|TKDE 2018|Similarity Metrics for SQL Query Clustering||
|TODS 2017|Fast and Accurate Time-Series Clustering||

### indexing
### MORE

|Source|Title|Classification|
|---|---|---|
|DASFAA-2018|Scalable Active Constrained Clustering for Temporal Data||
|ECML PKDD-2017|Identifying Representative Load Time Series for Load Flow Calculations||
|ICDM-2017|Distance and Density Clustering for Time Series Data||
|PODS-2018|Subtrajectory Clustering: Models and Algorithms	PODS-2018||
|SDM-2018|Interpretable Categorization of Heterogeneous Time Series Data||
|SIGIR-2018|CA-LSTM: Search Task Identification with Context Attention based LSTM||
|SIGKDD-2017|(Research Track最佳论文Runner Up)Toeplitz Inverse Covariance-Based Clustering of Multivariate Time Series Data||
|SIGKDD-2017|Effective and Real-time In-App Activity Analysis in Encrypted Internet Traffic Streams||
|SIGKDD-2017|Patient Subtyping via Time-Aware LSTM Networks||
|SIGKDD-2017|Robust Spectral Clustering for Noisy Data||
|SIGKDD-2017|Clustering Individual Transactional Data for Masses of Users||
|SIGKDD-2017|KATE: K-Competitive Autoencoder for Text||
|VLDB-2018|Clustering Stream Data by Exploring the Evolution of Density Mountain||
|VLDB-2017|Developing a Low Dimensional Patient Class Profile in Accordance to Their Respiration-Induced Tumor Motion||
|VLDB-2017|NG-DBSCAN: Scalable Density-Based Clustering for Arbitrary Data. 157-168||
|VLDB-2017|Local Search Methods for k-Means with Outliers. 757-768||
|VLDB-2017|Dimensions Based Data Clustering and Zone Maps. 1622-1633||
|VLDB-2015|YADING: Fast Clustering of Large-Scale Time Series Data. 473-484 ★★★||
|ICDE-2017|Density Based Clustering over Location Based Services. 461-469||
|ICDE-2017|A model-based approach for text clustering with outlier detection. 625-636||
|ICDE-2017|Streaming spectral clustering. 637-648||
|ICDE-2017|Accelerating large scale centroid-based clustering with locality sensitive hashing. 649-660||
|ICDE-2017|PurTreeClust: A purchase tree clustering algorithm for large-scale customer transaction data. 661-672||
|ICDE-2017|ClEveR: Clustering events with high density of true-to-false occurrence ratio. 918-929||

## Task 3 classfication (23+)

### representation

|Source|Title|Classification|
|---|---|---|
|TKDE 2016|Classifying Time Series Using Local Descriptors with Hybrid Sampling||
|TKDE 2015|Time-Series Classification with COTE: The Collective of Transformation-Based Ensembles.||
|TKDE 2014|Probabilistic Sequence Translation-Alignment Model for Time-Series Classification||

### similarity measure

### indexing

### MORE

|Source|Title|Classification|
|---|---|---|
|SIGKDD-2017|Effective and Real-time In-App Activity Analysis in Encrypted Internet Traffic Streams||
|ICDE-2017|ACTS: An Active Learning Method for Time Series Classification||
|ICDE-2017|Time Series Classification by Sequence Learning in All-Subsequence Space||
|VLDB-2017|Effects of Varying Sampling Frequency on the Analysis of Continuous ECG Data Streams||
|SDM-2018|Interpretable Categorization of Heterogeneous Time Series Data||
|SDM-2018|Evolving Separating References for Time Series Classification||
|SDM-2018|Classifying Multivariate Time Series by Learning Sequence-level Discriminative Patterns||
|SDM-2018|Brain EEG Time Series Selection: A Novel Graph-Based Approach for Classification||
|ICDM-2017|Linear Time Complexity Time Series Classification with Bag-of-Pattern-Features||
|EDBT-2018|Extracting Statistical Graph Features for Accurate and Efficient Time Series Classification||
|DASFAA-2018|Nearest Subspace with Discriminative Regularization for Time Series Classification||
|CIKM-2017|Fast and Accurate Time Series Classification with WEASEL||
|CIKM-2017|Does That Mean You're Happy?: RNN-based Modeling of User Interaction Sequences to Detect Good Abandonment||
|ECML PKDD-2017|Behavioral Constraint Template-Based Sequence Classification||
|ECML PKDD-2017|Cost Sensitive Time-Series Classification||
|ECML PKDD-2017|Efficient Temporal Kernels Between Feature Sets for Time Series Classification||
|ECML PKDD-2017|Analyzing Granger Causality in Climate Data with Time Series Classification Methods||
|ECML PKDD-2017|End-to-end Learning of Deep Spatio-temporal Representations for Satellite Image Time Series Classification||
|ECML PKDD-2017|Temporal and spatial approaches for land cover classification||
|ECML PKDD-2017|Self-Adaptive Ensemble Classifier for Handling Complex Concept Drift||


## Task 4 segmentation (3+)

### representation


|Source|Title|Classification|
|---|---|---|
|TKDE 2017|Efficient Pattern-Based Aggregation on Sequence Data.||
|TKDE 2014|An Adaptive Approach to Real-Time Aggregate Monitoring With Differential Privacy||

### similarity measure

### indexing

|Source|Title|Classification|
|---|---|---|
|TKDE 2018|BEATS: Blocks of Eigenvalues Algorithm for Time Series Segmentation.||



## Task 5 prediction (47+)

### representation

|Source|Title|Classification|
|---|---|---|
|NIPS 2018|Deep State Space Models for Time Series Forecasting||

### similarity measure

### indexing

### MORE

|Source|Title|Classification|
|---|---|---|
|SIGKDD-2017|Incremental Dual-memory LSTM in Land Cover Prediction||
|SIGKDD-2017|Mixture Factorized Ornstein-Uhlenbeck Processes for Time-Series Forecasting||
|SIGKDD-2017|Retrospective Higher-Order Markov Processes for User Trails||
|SIGKDD-2017|The Simpler The Better: A Unified Approach to Predicting Original Taxi Demands on Large-Scale Online Platforms||
|SIGKDD-2017|Stock Price Prediction via Discovering Multi-Frequency Trading Patterns||
|SIGKDD-2017|Dipole: Diagnosis Prediction in Healthcare via Attention-based Bidirectional Recurrent Neural Networks||
|SIGKDD-2017|Tracking the Dynamics in Crowdfunding||
|SIGKDD-2017|DeepMood: Modeling Mobile Phone Typing Dynamics for Mood Detection||
|SDM-2018|Sparse Decomposition for Time Series Forecasting and Anomaly Detection||
|SDM-2018|StreamCast: Fast and Online Mining of Power Grid Time Sequences||
|SDM-2018|Who will Attend This Event Together? Event Attendance Prediction via Deep LSTM Networks||
|SIGMOD-2018|Spatiotemporal Traffic Volume Estimation Model Based on GPS Samples||
|SIGMOD-2015|SMiLer: A Semi-Lazy Time Series Prediction System for Sensors||
|SIGIR-2018|A Flexible Forecasting Framework for Hiera1rchical Time Series with Seasonal Patterns: A Case Study of Web Traffic||
|SIGIR-2018|Modeling Long- and Short-Term Temporal Patterns with Deep Neural Networks||
|SIGIR-2018|Ad Click Prediction in Sequence with Long Short-Term Memory Networks: an Externality-aware Model||
|VLDB-2018|Forecasting Big Time Series: Old and New||
|VLDB-2018|Locality-Sensitive Hashing for Earthquake Detection: A Case Study Scaling Data-Driven Science||
|VLDB-2017|Matrix Profile IV: Using Weakly Lab1eled Time Series to Predict Outcomes||
|VLDB-2017|Flexible Online Task Assignment in Real-Time Spatial Data||
|VLDB-2017|A Time Machine for Information: Looking Back to Look Forward||
|ICDT-2018|Short-Term Traffic Forecasting: A Dynamic ST-KNN Model Considering Spatial Heterogeneity and Temporal Non-Stationarity||
|ICDM-2017|Spatio-Temporal Neural Networks for Space-Time Series Forecasting and Relations Discovery||
|ICDM-2017|Time-Aware Latent Hierarchical Model for Predicting House Prices||
|ICDM-2017|Autoregressive Tensor Factorization for Spatio-temporal Predictions||
|ICDM-2017|Deep and Confident Prediction for Time Series at Uber||
|ICDM-2017|Improving Multivariate Time Series Forecasting with Random Walks with Restarts on Causality Graphs||
|EDBT-2018|Big Data Analytics for Time Critical Maritime and Aerial Mobility Forecasting||
|DASFAA-2018|A Road-Aware Neural Network for Multi-step Vehicle Trajectory Prediction||
|CIKM-2017|Coupled Sparse Matrix Factorization for Response Time Prediction in Logistics Services||
|CIKM-2017|A Personalized Predictive Framework for Multivariate Clinical Time Series via Adaptive Model Selection||
|CIKM-2017|A Study of Feature Construction for Text-based Forecasting of Time Series Variables||
|CIKM-2017|Collaborative Sequence Prediction for Sequential Recommender||
|ECML PKDD-2017|BeatLex: Summarizing and Forecasting Time Series with Patterns||
|ECML PKDD-2017|Arbitrated Ensemble for Time Series Forecasting||
|ECML PKDD-2017|Forecasting and Granger Modelling with Non-linear Dynamical Dependencies||
|ECML PKDD-2017|PowerCast: Mining and Forecasting Power Grid Sequences||
|ECML PKDD-2017|Modeling the Temporal Nature of Human Behavior for Demographics Prediction||
|ECML PKDD-2017|Taking It for a Test Drive: A Hybrid Spatio-Temporal Model for Wildlife Poaching Prediction Evaluated Through a Controlled Field Test||
|ECML PKDD-2017|Predicting Defective Engines using Convolutional Neural Networks on Temporal Vibration Signals||
|ECML PKDD-2017|Usefulness of Unsupervised Ensemble Learning Methods for Time Series Forecasting of Aggregated or Clustered Load||
|ICDE-2017|Prediction-Based Task Assignment in Spatial Crowdsourcing. 997-1008||
|ICDE-2017|Discovering interpretable geo-social communities for user behavior prediction. 942-953||
|ICDE-2016|Link prediction in graph streams. 553-564||
|ICDE-2015|Searchlight: Context-aware predictive Continuous Querying of moving objects in symbolic space. 687-698||
|ICDE-2015|Predictive tree: An efficient index for predictive queries on road networks. 1215-1226||

## Task 6 anomaly detection (26+)

### survey

|Source|Title|Classification|
|---|---|---|
|TKDE 2014|Outlier Detection for Temporal Data: A Survey||

### representation

### similarity measure

|Source|Title|Classification|
|---|---|---|
|NIPS 2018|Precision and Recall for Time Series||

### indexing
### MORE

|Source|Title|Classification|
|---|---|---|
|SIGKDD-2017|Anomaly Detection in Streams with Extreme Value Theory||
|SIGKDD-2017|Let's See Your Digits: Anomalous-State Detection using Benford's Law||
|SIGKDD-2017|Finding Precursors to Anomalous Drop in Airspeed During a Flight's Take-off||
|SIGKDD-2017|Distributed Local Outlier Detection in Big Data||
|SIGKDD-2017|REMIX: Automated Exploration for Interactive Outlier Detection||
|SIGKDD-2017|Scalable Top-n Local Outlier Detection||
|SIGKDD-2017|Compass: Spatio Temporal Sentiment Analysis of US Election||
|SIGMOD-2018|TcpRT: Instrument and Diagnostic Analysis System for Service Quality of Cloud Databases at Massive Scale in Real-time||
|SIGMOD-2018|Auto-Detect: Data-Driven Error Detection in Tables||
|SIGMOD-2018|RushMon: Real-time Isolation Anomalies Monitoring||
|ICDE-2018|Improving Quality of Observational Streaming Medical Data by Using Long Short-Term Memory Networks (LSTMs)||
|ICDE-2017|LSHiForest: A Generic Framework for Fast Tree Isolation Based Ensemble Anomaly Analysis. 983-994||
|VLDB-2017|Time Series Data Cleaning: From Anomaly Detection to Anomaly Repairing||
|VLDB-2017|Local Search Methods for k-Means with Outliers||
|VLDB-2016|Streaming Anomaly Detection Using Randomized Matrix Sketching. 192-203 ★★★||
|SDM-2018|Sparse Decomposition for Time Series Forecasting and Anomaly Detection||
|SDM-2018|StreamCast: Fast and Online Mining of Power Grid Time Sequences||
|SDM-2018|Outlier Detection over Distributed Trajectory Streams||
|ICDM-2017|Matrix Profile VII: Time Series Chains: A New Primitive for Time Series Data Mining (Best Student Paper Award)||
|ICDM-2017|Dependency Anomaly Detection for Heterogeneous Time Series: A Granger-Lasso Approach||
|CIKM-2017|Efficient Discovery of Abnormal Event Sequences in Enterprise Security Systems||
|CIKM-2017|Anomaly Detection in Dynamic Networks using Multi-view Time-Series Hypersphere Learning||
|ECML PKDD-2017|UAPD: Predicting Urban Anomalies from Spatial-Temporal Data||
|ECML PKDD-2017|Transfer Learning for Time Series Anomaly Detection||

## Task 7 motif discovery (10+)

### representation

### similarity measure

### indexing

### MORE
|Source|Title|Classification|
|---|---|---|
|SIGKDD-2017|Matrix Profile V: A Generic Technique to  Incorporate Domain Knowledge into Motif Discovery ||
|SIGKDD-2017|Contextual Motifs: Increasing the Utility of Motifs using Contextual Data||
|SIGMOD-2018|Matrix Profile X: VALMOD - Scalable Discovery of Variable-Length Motifs in Data Series||
|SIGMOD-2018|VALMOD: A Suite for Easy and Exact Detection of Variable Length Motifs in Data Series||
|ICDM-2017|IterativE Grammar-Based Framework for Discovering Variable-Length Time Series Motifs||
|ICDM-2017|Efficient discovery of time series motifs with large length range in million scale time series||
|ICDM-2017|Matrix Profile VI: Meaningful Multidimensional Motif Discovery||
|ICDE-2016|Fast motif discovery in short sequences. 1158-1169||
|ICDE-2015|Quick-motif: An efficient and scalable framework for exact motif discovery. 579-590||
|VLDB-2015|Rare Time Series Motif Discovery from Unbounded Streams. 149-160||

## Task 8 analysis (2+)

### representation

### similarity measure

|Source|Title|Classification|
|---|---|---|
|TKDE 2018|Diverse Relevance Feedback for Time Series with Autoencoder Based Summarizations||

### indexing

|Source|Title|Classification|
|---|---|---|
|TKDE 2017|Time Series Management Systems: A Survey.||
