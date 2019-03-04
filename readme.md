已有从论文[Timeseries data mining](https://dl.acm.org/citation.cfm?id=2379788)（2012）之后，2013-2018年的重要会议的时间序列相关论文列表（见下文Paper List）。

接下来，快速阅读每篇论文的Abstract和Introduction，按照“新问题”和“新方法”对论文进行分类。
其中新方法的论文暂时放一边，重点关注新问题，并记录2013-2018年间产生的新问题。
* 新问题关注度 > 新方法关注度
* 提出新问题的论文的工作量<提出新方法的论文的工作量，因为后者需要battle所有已有的方法

此外，了解Introduction的典型结构有助于快速阅读：
1. 大量的时间序列产生
2. 在工业时间序列中 工况需要分段
3. 现在是人工来做这件事，也有一些其它方法，但是问题缺陷
4. 这件事情non-trivial 有难度
5. 我们的方法怎么对应上面的non-trivial 一些结论 在数据集上验证
6. 总结我们的contributions
7. 后文的结构

# 分工
芮：prediction(47+)

康: anomaly detection(26+), motif discovery(10+), analysis(2+)

江: query by content(16+), classfication(23+)

安: clustering(28+), segmentation(3+)

# 2013-2019新问题（under construction）
1. 

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

* TKDE 2016
  Metric All-k-Nearest-Neighbor Search. 示例：新问题
* NIPS 2017
  Soft-DTW: a Differentiable Loss Function for Time-Series
* NIPS 2018
  Autowarp: Learning a Warping Distance from Unlabeled Time Series Using Sequence Autoencoders

### indexing

### MORE

* SIGIR-2018
  CA-LSTM: Search Task Identification with Context Attention based LSTM

* SIGMOD-2018
  Qetch: Time Series Querying with Expressive Sketches

* SIGMOD-2017
  1. Approximate Query Processing: No Silver Bullet
  2. Approximate Query Engines: Commercial Challenges and Research Opportunities
  3. Approximate Query Processing for Interactive Data Science

* VLDB-2017
  DITIR: Distributed Index for High Throughput Trajectory Insertion and Real-time Temporal Range Query

* DASFAA-2018
  Time-Based Trajectory Data Partitioning for Efficient Range Query

* ICDE-2017
  Tracking Matrix Approximation over Distributed Sliding Windows. 833-844

* ICDE-2015
  Predictive tree: An efficient index for predictive queries on road networks. 1215-1226

* TKDE 2017
  Measuring Concentration of Distances—An Effective and Efficient Empirical Index.

* TKDE 2018
    1. Non-Overlapping Subsequence Matching of Stream Synopses
    2. Reverse k Nearest Neighbor Search over Trajectories
    3. Fast Cosine Similarity Search in Binary Space with Angular Multi-Index Hashing

## Task 2 clustering (28+)

### survey

* TODS 2013
  Data Stream Clustering: A Survey

### representation

### similarity measure

* TKDE 2016 
  Similarity Measure Selection for Clustering Time Series Databases.
* TKDE 2018
  Similarity Metrics for SQL Query Clustering
* TODS 2017
  Fast and Accurate Time-Series Clustering

### indexing
### MORE
* DASFAA-2018
Scalable Active Constrained Clustering for Temporal Data

* ECML PKDD-2017
Identifying Representative Load Time Series for Load Flow Calculations

* ICDM-2017
Distance and Density Clustering for Time Series Data

* PODS-2018
Subtrajectory Clustering: Models and Algorithms	PODS-2018

* SDM-2018
Interpretable Categorization of Heterogeneous Time Series Data

* SIGIR-2018
CA-LSTM: Search Task Identification with Context Attention based LSTM

* SIGKDD-2017
    1. （Research Track最佳论文Runner Up）Toeplitz Inverse Covariance-Based Clustering of Multivariate Time Series Data
    2. Effective and Real-time In-App Activity Analysis in Encrypted Internet Traffic Streams
    3. Patient Subtyping via Time-Aware LSTM Networks
    4. Robust Spectral Clustering for Noisy Data
    5. Clustering Individual Transactional Data for Masses of Users
    6. KATE: K-Competitive Autoencoder for Text

* VLDB-2018
Clustering Stream Data by Exploring the Evolution of Density Mountain

* VLDB-2017 
    1. Developing a Low Dimensional Patient Class Profile in Accordance to Their Respiration-Induced Tumor Motion
    2. NG-DBSCAN: Scalable Density-Based Clustering for Arbitrary Data. 157-168
    3. Local Search Methods for k-Means with Outliers. 757-768
    4. Dimensions Based Data Clustering and Zone Maps. 1622-1633

* VLDB-2015
YADING: Fast Clustering of Large-Scale Time Series Data. 473-484 ★★★

* ICDE-2017
    1. Density Based Clustering over Location Based Services. 461-469
    2. A model-based approach for text clustering with outlier detection. 625-636
    3. Streaming spectral clustering. 637-648
    4. Accelerating large scale centroid-based clustering with locality sensitive hashing. 649-660
    5. PurTreeClust: A purchase tree clustering algorithm for large-scale customer transaction data. 661-672
    6. ClEveR: Clustering events with high density of true-to-false occurrence ratio. 918-929

## Task 3 classfication (23+)

### representation

* TKDE 2016
  Classifying Time Series Using Local Descriptors with Hybrid Sampling

* TKDE 2015
  Time-Series Classification with COTE: The Collective of Transformation-Based Ensembles.

* TKDE 2014

  Probabilistic Sequence Translation-Alignment Model for Time-Series Classification

### similarity measure

### indexing

### MORE
* SIGKDD-2017
Effective and Real-time In-App Activity Analysis in Encrypted Internet Traffic Streams

* ICDE-2017
    1. ACTS: An Active Learning Method for Time Series Classification
    2. Time Series Classification by Sequence Learning in All-Subsequence Space

* VLDB-2017
Effects of Varying Sampling Frequency on the Analysis of Continuous ECG Data Streams

* SDM-2018
    1. Interpretable Categorization of Heterogeneous Time Series Data
    2. Evolving Separating References for Time Series Classification
    3. Classifying Multivariate Time Series by Learning Sequence-level Discriminative Patterns
    4. Brain EEG Time Series Selection: A Novel Graph-Based Approach for Classification

* ICDM-2017
Linear Time Complexity Time Series Classification with Bag-of-Pattern-Features

* EDBT-2018
Extracting Statistical Graph Features for Accurate and Efficient Time Series Classification

* DASFAA-2018
Nearest Subspace with Discriminative Regularization for Time Series Classification

* CIKM-2017
    1. Fast and Accurate Time Series Classification with WEASEL
    2. Does That Mean You're Happy?: RNN-based Modeling of User Interaction Sequences to Detect Good Abandonment

* ECML PKDD-2017
    1. Behavioral Constraint Template-Based Sequence Classification
    2. Cost Sensitive Time-Series Classification
    3. Efficient Temporal Kernels Between Feature Sets for Time Series Classification
    4. Analyzing Granger Causality in Climate Data with Time Series Classification Methods
    5. End-to-end Learning of Deep Spatio-temporal Representations for Satellite Image Time Series Classification
    6. Temporal and spatial approaches for land cover classification
    7. Self-Adaptive Ensemble Classifier for Handling Complex Concept Drift


## Task 4 segmentation (3+)

### representation

* TKDE 2017
  Efficient Pattern-Based Aggregation on Sequence Data.
* TKDE 2014
  An Adaptive Approach to Real-Time Aggregate Monitoring With Differential Privacy

### similarity measure

### indexing

* TKDE 2018

  BEATS: Blocks of Eigenvalues Algorithm for Time Series Segmentation.



## Task 5 prediction (47+)

### representation

* NIPS 2018
  Deep State Space Models for Time Series Forecasting

### similarity measure

### indexing

### MORE

* SIGKDD-2017
    1. Incremental Dual-memory LSTM in Land Cover Prediction
    2. Mixture Factorized Ornstein-Uhlenbeck Processes for Time-Series Forecasting
    3. Retrospective Higher-Order Markov Processes for User Trails
    4. The Simpler The Better: A Unified Approach to Predicting Original Taxi Demands on Large-Scale Online Platforms
    5. Stock Price Prediction via Discovering Multi-Frequency Trading Patterns
    6. Dipole: Diagnosis Prediction in Healthcare via Attention-based Bidirectional Recurrent Neural Networks
    7. Tracking the Dynamics in Crowdfunding
    8. DeepMood: Modeling Mobile Phone Typing Dynamics for Mood Detection


* SDM-2018
    1. Sparse Decomposition for Time Series Forecasting and Anomaly Detection
    2. StreamCast: Fast and Online Mining of Power Grid Time Sequences
    3. Who will Attend This Event Together? Event Attendance Prediction via Deep LSTM Networks


* SIGMOD-2018
Spatiotemporal Traffic Volume Estimation Model Based on GPS Samples

* SIGMOD-2015
SMiLer: A Semi-Lazy Time Series Prediction System for Sensors


* SIGIR-2018
    1. A Flexible Forecasting Framework for Hiera1rchical Time Series with Seasonal Patterns: A Case Study of Web Traffic
    2. Modeling Long- and Short-Term Temporal Patterns with Deep Neural Networks
    3. Ad Click Prediction in Sequence with Long Short-Term Memory Networks: an Externality-aware Model

* VLDB-2018
    1. Forecasting Big Time Series: Old and New
    2. Locality-Sensitive Hashing for Earthquake Detection: A Case Study Scaling Data-Driven Science

* VLDB-2017
    1. Matrix Profile IV: Using Weakly Lab1eled Time Series to Predict Outcomes
    2. Flexible Online Task Assignment in Real-Time Spatial Data
    3. A Time Machine for Information: Looking Back to Look Forward


* ICDT-2018
Short-Term Traffic Forecasting: A Dynamic ST-KNN Model Considering Spatial Heterogeneity and Temporal Non-Stationarity

* ICDM-2017
    1. Spatio-Temporal Neural Networks for Space-Time Series Forecasting and Relations Discovery
    2. Time-Aware Latent Hierarchical Model for Predicting House Prices
    3. Autoregressive Tensor Factorization for Spatio-temporal Predictions
    4. Deep and Confident Prediction for Time Series at Uber
    5. Improving Multivariate Time Series Forecasting with Random Walks with Restarts on Causality Graphs

* EDBT-2018
Big Data Analytics for Time Critical Maritime and Aerial Mobility Forecasting

* DASFAA-2018
A Road-Aware Neural Network for Multi-step Vehicle Trajectory Prediction

* CIKM-2017
    1. Coupled Sparse Matrix Factorization for Response Time Prediction in Logistics Services
    2. A Personalized Predictive Framework for Multivariate Clinical Time Series via Adaptive Model Selection
    3. A Study of Feature Construction for Text-based Forecasting of Time Series Variables
    4. Collaborative Sequence Prediction for Sequential Recommender

* ECML PKDD-2017
    1. BeatLex: Summarizing and Forecasting Time Series with Patterns
    2. Arbitrated Ensemble for Time Series Forecasting
    3. Forecasting and Granger Modelling with Non-linear Dynamical Dependencies
    4. PowerCast: Mining and Forecasting Power Grid Sequences
    5. Modeling the Temporal Nature of Human Behavior for Demographics Prediction
    6. Taking It for a Test Drive: A Hybrid Spatio-Temporal Model for Wildlife Poaching Prediction Evaluated Through a Controlled Field Test
    7. Predicting Defective Engines using Convolutional Neural Networks on Temporal Vibration Signals
    8. Usefulness of Unsupervised Ensemble Learning Methods for Time Series Forecasting of Aggregated or Clustered Load

* ICDE-2017
    1. Prediction-Based Task Assignment in Spatial Crowdsourcing. 997-1008
    2. Discovering interpretable geo-social communities for user behavior prediction. 942-953

* ICDE-2016
Link prediction in graph streams. 553-564

* ICDE-2015
    1. Searchlight: Context-aware predictive Continuous Querying of moving objects in symbolic space. 687-698
    2. Predictive tree: An efficient index for predictive queries on road networks. 1215-1226

## Task 6 anomaly detection (26+)

### survey

* TKDE 2014
  Outlier Detection for Temporal Data: A Survey

### representation

### similarity measure

* NIPS 2018
  Precision and Recall for Time Series

### indexing
### MORE
* SIGKDD-2017
    1. Anomaly Detection in Streams with Extreme Value Theory
    2. Let's See Your Digits: Anomalous-State Detection using Benford's Law
    3. Finding Precursors to Anomalous Drop in Airspeed During a Flight's Take-off
    4. Distributed Local Outlier Detection in Big Data
    5. REMIX: Automated Exploration for Interactive Outlier Detection
    6. Scalable Top-n Local Outlier Detection
    7. Compass: Spatio Temporal Sentiment Analysis of US Election

* SIGMOD-2018
    1. TcpRT: Instrument and Diagnostic Analysis System for Service Quality of Cloud Databases at Massive Scale in Real-time
    2. Auto-Detect: Data-Driven Error Detection in Tables
    3. RushMon: Real-time Isolation Anomalies Monitoring

* ICDE-2018
Improving Quality of Observational Streaming Medical Data by Using Long Short-Term Memory Networks (LSTMs)

* ICDE-2017
LSHiForest: A Generic Framework for Fast Tree Isolation Based Ensemble Anomaly Analysis. 983-994


* VLDB-2017
    1. Time Series Data Cleaning: From Anomaly Detection to Anomaly Repairing
    2. Local Search Methods for k-Means with Outliers

* VLDB-2016
Streaming Anomaly Detection Using Randomized Matrix Sketching. 192-203 ★★★


* SDM-2018
    1. Sparse Decomposition for Time Series Forecasting and Anomaly Detection
    2. StreamCast: Fast and Online Mining of Power Grid Time Sequences
    3. Outlier Detection over Distributed Trajectory Streams

* ICDM-2017
    1. Matrix Profile VII: Time Series Chains: A New Primitive for Time Series Data Mining (Best Student Paper Award)
    2. Dependency Anomaly Detection for Heterogeneous Time Series: A Granger-Lasso Approach

* CIKM-2017
    1. Efficient Discovery of Abnormal Event Sequences in Enterprise Security Systems
    2. Anomaly Detection in Dynamic Networks using Multi-view Time-Series Hypersphere Learning

* ECML PKDD-2017
    1. UAPD: Predicting Urban Anomalies from Spatial-Temporal Data
    2. Transfer Learning for Time Series Anomaly Detection

## Task 7 motif discovery (10+)

### representation

### similarity measure

### indexing

### MORE
* SIGKDD-2017
    1. Matrix Profile V: A Generic Technique to  Incorporate Domain Knowledge into Motif Discovery 
    2. Contextual Motifs: Increasing the Utility of Motifs using Contextual Data

* SIGMOD-2018
    1. Matrix Profile X: VALMOD - Scalable Discovery of Variable-Length Motifs in Data Series
    2. VALMOD: A Suite for Easy and Exact Detection of Variable Length Motifs in Data Series

* ICDM-2017
    1. IterativE Grammar-Based Framework for Discovering Variable-Length Time Series Motifs
    2. Efficient discovery of time series motifs with large length range in million scale time series
    3. Matrix Profile VI: Meaningful Multidimensional Motif Discovery

* ICDE-2016
Fast motif discovery in short sequences. 1158-1169

* ICDE-2015
Quick-motif: An efficient and scalable framework for exact motif discovery. 579-590

* VLDB-2015
Rare Time Series Motif Discovery from Unbounded Streams. 149-160

## Task 8 analysis (2+)

### representation

### similarity measure

* TKDE 2018
  Diverse Relevance Feedback for Time Series with Autoencoder Based Summarizations

### indexing

* TKDE 2017
  Time Series Management Systems: A Survey.
