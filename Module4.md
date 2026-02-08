# Module 4: Get Started with Machine Learning in Azure (1000 XP)
## Define the problem
This is done by understanding
- What the models output should be
- Type of machine learning task 
- Crietria making the model successful

Machine learning tasks
- Classification: Predict a category value
- Regression: Predict a numeric value
- Time-series forecasting: Predict a value at a future time
- Computer vision: Analyze images and detect objects in images
- natural language processing: Analyze text and extract meaning

Scenario flow
1. Load data: import and inspect dataset
2. Preprocess data: normalize and clean for consistency
3. Split data: seperate data into training and validation sets
4. Choose model: select and configure algorithm
5. Train model: Learn patterns from the data
6. Score model: generate predictions on test data
7. Evaluate model: compare predicted values to actual values
## Get and prepare data
To train a model you need to 
- identify data source and format
- choose how to serve the data
- design a data ingestion solution

Data ingestion
- Extract, transform, load (ETL)
- Data ingestion pipeline can be used to move and transform data
- Data ingestion pipeline is a sequence of tasks that move and transform data
- Such pipelines can be created using azure services like
    - Azure synapse analytics
    - Azure Databricks
    - Azure Machine learning
- common approach for data ingestion
    1. Extract raw data
    2. Copy and transform the data with Azure synapse analytics
    3. Store the prepared data in an Azure Blob storage 
    4. Train the model with Azure Machine learning



## Train the model
Services
- Azure Machine Learning
    - Gives many options to train and manage your ml models.
    - Can choose to  work with the studio for a UI experience or manage the ml workloads with python SDK or CLI for code-first experience
- Azure Databricks
    - Data analytics platform that can you can use for data engineering and data science.
    Uses distibuted spark compute to efficiently process large datasets.
    - YOu can train & manage models with Azure databricks or integrate Azure databricks with other services such as Azure machine learning
- Azure Fabric
    - Integrated analytics plafrom designed to streamline data workflows between data analysitcs, engineer and scienctists 
    - you can prepare data, train model, use the trained model to fenerate predictions, and visualize the data in power bi reports
- Foundry tools
    - Collection of prebuilt ml models that can be used for common ml taks ect object detect in images
    - Modles offered as an API which can intefrate with my application
    - Some models can be customized with own training data 

### Azure Machine learning
A cloud seervice for training, deploying and managing ml models. its taks include
- Exploring data and preparing it for modeling
- Trianing and evaluating ml models
- Registering and managing trained models
- Deploying trained models for use by apps and services
- Reviewing and applyign repsonsible AI principles

#### Features
- Centralized storage and management of datasets for model training and evaluation.
- On-demand compute resources on which you can run machine learning jobs, such as training a model.
- **Automated machine learning (AutoML)**, which makes it easy to run multiple training jobs with different algorithms and parameters to find the best model for your data.
- Visual tools to define orchestrated pipelines for processes such as model training or inferencing.
- Integration with common machine learning frameworks such as MLflow, which make it easier to manage model training, evaluation, and deployment at scale.
- Built-in support for visualizing and evaluating metrics for responsible AI, including model explainability, fairness assessment, and others.

## Use Azure Machine Learning studio
Azure Machine Learning studio is a web-based interface for training and managing ml models.
You can
- Import and explore data.
- Create and use compute resources.
- Run code in notebooks.
- Use visual tools to create jobs and pipelines.
- Use automated machine learning to train models.
- View details of trained models, including evaluation metrics, responsible AI information, and training parameters.
- Deploy trained models for on-request and batch inferencing.
- Import and manage models from a comprehensive model catalog.

Deciding between compute options
Monitor how long it taks to train the model and how much compute is used to execute the code
- CPU: less expensive but slower, for smaller tabular datasets
- GPU: more expensive but faster, for larger tabular datasets and unstructured data like images and text

### Azure automated machine learning
You are automatically assigned to a compute resource to run the training job. 
You have access to your own datasets and your trained ml models can be deployed as services

## Integrate a model
To integrate the model, you need to deploy a model to an endpoint. You can deploy a model to an endpoint for either real-time or batch predictions.

### Deploy a model to an endpoint
When you train a model, the goal is often to integrate the model into an application.
- To easily integrate a model into an application, you can use endpoints. 
- An endpoint can be a web address that an application can call to get a message back.

When you deploy a model to an endpoint, you have two options:
- Get real-time predictions
- Get batch predictions

#### Get real-time predictions
If you want the model to score any new data as it comes in, you need predictions in real-time. Real-time predictions are often needed when a model is used by an application such as a mobile app or a website.

Imagine you have a website that contains your product catalog:

A customer selects a product on your website. Based on the customer's selection, the model recommends other items from the product catalog immediately. 

#### Get batch predictions
If you want the model to score new data in batches, and save the results as a file or in a database, you need batch predictions.

For example, you can train a model that predicts orange juice sales for each future week. By predicting orange juice sales, you can ensure that supply is sufficient to meet expected demand.

Imagine you're visualizing all historical sales data in a report. You'll want to include the predicted sales in the same report.

A collection of data points is referred to as a batch.

### Decide between real-time or batch deployment
To decide whether to design a real-time or batch deployment solution, you need to consider the following questions:
- How often should predictions be generated?
- How soon are the results needed?
- Should predictions be generated individually or in batches?
- How much compute power is needed to execute the model?

#### Identify the necessary frequency of scoring
A common scenario is that you're using a model to score new data. Before you can get predictions in real-time or in batch, you must first collect the new data.

There are various ways to generate or collect data. New data can also be collected at different time intervals.

For example, you can extract financial data from a database every three months.

Generally, there are two types of use cases:
- You need the model to score the new data as soon as it comes in.
- You can schedule or trigger the model to score the new data that you've collected over time.

Whether you want real-time or batch predictions doesn't necessarily depend on how often new data is collected. Instead, it depends on how often and how quickly you need the predictions to be generated.

- If you need the model's predictions immediately when new data is collected, you need real-time predictions. 
- If the model's predictions are only consumed at certain times, you need batch predictions.

#### Decide on the number of predictions
Another important question to ask yourself is whether you need the predictions to be generated individually or in batches.

A simple way to illustrate the difference between individual and batch predictions is to imagine a table. Suppose you have a table of customer data where each row represents a customer. For each customer, you have some demographic data and behavioral data, such as how many products they've purchased from your web shop and when their last purchase was.

Based on this data, you can predict customer churn

Once you've trained the model, you can decide if you want to generate predictions:

- Individually: The model receives a single row of data and returns whether or not that individual customer will buy again.
- Batch: The model receives multiple rows of data in one table and returns whether or not each customer will buy again. The results are collated in a table that contains all predictions.

You can also generate individual or batch predictions when working with files. For example, when working with a computer vision model you may need to score a single image individually, or a collection of images in one batch.

#### Consider the cost of compute
In addition to using compute when training a model, you also need compute when deploying a model.. To decide whether to deploy your model to a real-time or batch endpoint, you must consider the cost of each type of compute.
- If you need real-time predictions, you need compute that is always available and able to return the results (almost) immediately. 
    - Container technologies like Azure Container Instance (ACI) and Azure Kubernetes Service (AKS) are ideal for such scenarios as they provide a lightweight infrastructure for your deployed model.
- However, when you deploy a model to a real-time endpoint and use such container technology, the compute is always on. Once a model is deployed, you're continuously paying for the compute as you can't pause, or stop the compute as the model must always be available for immediate predictions.
- If you need batch predictions, you need compute that can handle a large workload. Ideally, you'd use a compute cluster that can score the data in parallel batches by using multiple nodes.
    - When working with compute clusters that can process data in parallel batches, the compute is provisioned by the workspace when the batch scoring is triggered, and scaled down to 0 nodes when there's no new data to process. By letting the workspace scale down an idle compute cluster, you can save significant costs.
### Summary
