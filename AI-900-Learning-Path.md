# AI-900: Azure AI Fundamentals Learning Path

## Table of Contents

- [Module 1: Overview of AI Concepts (1100 XP)](#module-1-overview-of-ai-concepts-1100-xp)
- [Module 2: Get Started in Microsoft Foundry (1000 XP)](#module-2-get-started-in-microsoft-foundry-1000-xp)
- [Module 3: Introduction to Machine Learning Concepts (1200 XP)](#module-3-introduction-to-machine-learning-concepts-1200-xp)
- [Module 4: Get Started with Machine Learning in Azure (1000 XP)](#module-4-get-started-with-machine-learning-in-azure-1000-xp)
- [Module 5: Introduction to Generative AI and Agents (1000 XP)](#module-5-introduction-to-generative-ai-and-agents-1000-xp)
- [Module 6: Get Started with Generative AI and Agents in Microsoft Foundry (800 XP)](#module-6-get-started-with-generative-ai-and-agents-in-microsoft-foundry-800-xp)
- [Module 7: Introduction to Text Analysis Concepts (1000 XP)](#module-7-introduction-to-text-analysis-concepts-1000-xp)
- [Module 8: Get Started with Text Analysis in Microsoft Foundry (800 XP)](#module-8-get-started-with-text-analysis-in-microsoft-foundry-800-xp)
- [Module 9: Introduction to AI Speech Concepts (800 XP)](#module-9-introduction-to-ai-speech-concepts-800-xp)
- [Module 10: Get Started with Speech in Microsoft Foundry (1000 XP)](#module-10-get-started-with-speech-in-microsoft-foundry-1000-xp)
- [Module 11: Introduction to Computer Vision Concepts (900 XP)](#module-11-introduction-to-computer-vision-concepts-900-xp)
- [Module 12: Get Started with Computer Vision in Microsoft Foundry (800 XP)](#module-12-get-started-with-computer-vision-in-microsoft-foundry-800-xp)
- [Module 13: Introduction to AI-Powered Information Extraction Concepts (1000 XP)](#module-13-introduction-to-ai-powered-information-extraction-concepts-1000-xp)
- [Module 14: Get Started with AI-Powered Information Extraction in Microsoft Foundry (1000 XP)](#module-14-get-started-with-ai-powered-information-extraction-in-microsoft-foundry-1000-xp)

---

# Module 1: Overview of AI Concepts (1100 XP)
## Generative AI and agents
**Key Concepts:**
- **Generative AI**: Branch of AI that enables software applications to generate new content
- **Language Models**: Trained with huge volumes of data - often documents from the Internet or other public sources of information
- **Prompts**: Natural language statements or questions that users interact with to initiate generation of meaningful responses
- **Semantic Relationships**: Models "know" how words relate to one another, enabling generation of meaningful text sequences
- **Large Language Models (LLMs)**: Powerful, generalize well, but more costly to train and use
- **Small Language Models (SLMs)**: Work well for specific topic areas, easily deployed for local applications and device agents

## Text and natural language
**Key Concepts:**
- **Natural Language Processing (NLP)**: Techniques for making sense of language
- **Foundation for LLMs**: NLP is the foundation on which generative AI large language models (LLMs) are built

**Text Analysis Capabilities:**
- **Text Classification**: Assigning document to a specific category; including sentiment analysis
- **Key-Term Extraction and Entity Detection**: Identifying key words or phrases in a document
- **Summarization**: Reducing the volume of text while still encapsulating the main points

**Text Analysis Scenarios:**
- Analyzing document or transcripts of calls and meetings to determine key subjects and identify specific mentions of people, places, organizations, products, or other entities
- Analyzing social media posts, product reviews, or articles to evaluate sentiment and opinion
- Implementing chatbots that can answer frequently asked questions or orchestrate predictable conversational dialogs that don't require the complexity of generative AI

## Speech
**Key Concepts:**
- **Speech Capabilities**: Enable users to interact with AI applications and agents through spoken language
- **Speech Recognition**: The ability of AI to "hear" and interpret speech; (speech-to-text) where the audio signal for the speech is transcribed into text
- **Speech Synthesis**: The ability of AI to vocalize words as spoken language; (text-to-speech) in which information in text format is converted into an audible signal
- **Evolving Technology**: AI speech technology is evolving rapidly to handle challenges like ignoring background noise, detecting interruptions, and generating increasingly expressive and human-like voices

**AI Speech Scenarios:**
- AI agents that understand spoken input, perform tasks, and respond with spoken results
- Automated transcription of calls or meetings
- Automating audio descriptions of video or text
- Combined speech technologies enable natural voice interactions

## Computer vision
**Key Concepts:**
- **Computer Vision**: Area of artificial intelligence that deals with the analysis of visual input.
- **Training Process**: Computer vision is accomplished by using large numbers of images to train a model

**Types of Computer Vision Models:**
- **Image Classification**: Model is trained with images that are labeled with the main subject of the image so it can analyze unlabeled images and predict the most appropriate label - identifying the subject of the image
- **Object Detection**: Model is trained to identify the location of specific objects in an image
- **Semantic Segmentation**: Advanced form of object detection where, rather than indicate an object's location by drawing a box around it, the model can identify the individual pixels in the image that belong to a particular object
- **Multi-modal Models**: Combine visual features and associated text descriptions, enabling them to generate comprehensive descriptions of images

**Computer Vision Scenarios:**
- AI agents that can interpret visual input
- Auto-captioning or tag-generation for photographs
- Visual search
- Monitoring stock levels or identifying items for checkout in retail scenarios
- Security video monitoring
- Authentication through facial recognition
- Robotics and self-driving vehicles

## Information extraction
**Key Concepts:**
- **Information Extraction**: Finding information and unlocking insights in unstructured data sources, such as scanned documents and forms, images, and audio or video recordings
- **Optical Character Recognition (OCR)**: Computer vision technology that can identify the location of text in an image
- **Field Extraction**: OCR is often combined with an analytical model that can interpret individual values in the document, and so extract specific fields
- **Advanced Models**: More advanced models that can extract information from audio recording, images, and videos are becoming more readily available

**Data and Insight Extraction Scenarios:**
- Automated processing of forms and other documents in a business process - for example, processing an expense claim
- Large-scale digitization of data from paper forms. For example, scanning and archiving census records
- Identifying key points and follow-up actions from meeting transcripts or recordings

## Responsible AI
**Principles for responsible AI include:**

**Fairness**
- AI models are trained using data, which is generally sourced and selected by humans
- There's substantial risk that the data selection criteria, or the data itself reflects unconscious bias that may cause a model to produce discriminatory outputs
- AI developers need to take care to minimize bias in training data and test AI systems for fairness

**Reliability and safety**
- AI is based on probabilistic models
- AI-powered applications need to take this into account and mitigate risks accordingly

**Privacy and security**
- Models are trained using data, which may include personal information
- AI developers have a responsibility to ensure that the training data is kept secure, and that the trained models themselves can't be used to reveal private personal.

**Inclusiveness**
- The potential of AI to improve lives and drive success should be open to everyone
- AI developers should strive to ensure that their solutions don't exclude some users

**Transparency**
- It's important to make users aware of how the system works and any potential limitations it may have

**Accountability**
- It's important for organizations developing AI models and applications to define and apply a framework of governance to help ensure that they apply responsible AI principles to their work

**Responsible AI Examples:**
- **College Admissions**: AI-powered college admissions system should be tested to ensure it evaluates all applications fairly, taking into account relevant academic criteria but avoiding unfounded discrimination based on irrelevant demographic factors
- **Facial Recognition**: Facial identification system used in an airport or other secure area should delete personal images that are used for temporary access as soon as they're no longer required. 
- **Accessibility**: AI agent that offers speech-based interaction should also generate text captions to avoid making the system unusable for users with a hearing impairment
- **Disclosure**: Bank that uses an AI-based loan-approval application should disclose the use of AI.

## Summary
**Module 1 Key Takeaways:**
- **AI Fundamentals**: AI encompasses multiple technologies that simulate human intelligence, including generative AI that creates new content and traditional AI that analyzes existing data
- **Core AI Capabilities**: Text and natural language processing, speech recognition and synthesis, computer vision, and information extraction enable machines to understand and interact with the world
- **Generative AI**: Uses language models trained on vast data to generate content through prompts, with LLMs offering broad capabilities and SLMs providing focused, efficient solutions
- **Responsible AI**: Essential principles of fairness, reliability, privacy, inclusiveness, transparency, and accountability ensure ethical and trustworthy AI development and deployment

# Module 2: Get Started in Microsoft Foundry (1000 XP)
## What is an AI application?
**Artificial Intelligence**: Systems designed to perform tasks that typically require human intelligence

**Responsible AI**: Emphasizes fairness, transparency, and ethical use of AI technologies.

### Key AI workloads:

- **Generative AI**: Create
- **Agents and automation**: Automate
- **Speech**: Listen
- **Text analysis**: Read
- **Computer Vision**: See
- **Information Extraction**: Extract

**Machine learning**: Enables machines to learn patterns from data and improve performance without explicit programming.

### Types of ML:

- **Supervised**: Regression for predicting prices, classification for spam detection.
- **Unsupervised**: Clustering for customer segmentation.
- **Deep Learning**: Using neural networks with multiple layers for tasks like image recognition and speech synthesis. 
- **Generative AI**: uses deep learning capabilities to create new content—text, images, audio, code—rather than just classify or predict outcomes.

**AI vs ML**: AI is the broad goal of creating intelligent systems that mimic human intelligence. ML is the primary method to achieve AI, using data-driven algorithms to enable machines to learn patterns from data and improve without explicit programming.



### AI Applications

- **Model-powered**: They use trained models to process inputs and generate outputs, such as text, images, or decisions.
- **Dynamic**: Unlike static programs, AI apps can improve over time through retraining or fine-tuning.

Some of the typical ways people interact with AI applications include:

- **Conversational Interfaces**: Users interact via chatbots or voice assistants (such as: asking questions, getting recommendations).
- **Embedded Features**: AI is integrated into apps for tasks like autocomplete, image recognition, or fraud detection.
- **Decision Support**: AI applications provide insights or predictions to help users make informed choices (such as: personalized shopping, medical diagnostics).
- **Automation**: They handle repetitive tasks, such as document processing or customer service, reducing manual effort.

## Components of an AI application
Microsoft supports each layer of an AI application: the data layer, model layer, compute layer, and orchestration layer.

### Data Layer
- The data layer is the foundation of any AI application.
- It includes the collection, storage, and management of data used for training, inference, and decision-making.
- Common data sources include structured databases such as Azure SQL and PostgreSQL, unstructured data, such as documents and images, and real-time streams. 
- Azure services like Cosmos DB and Azure Data Lake are often used to store and manage large-scale datasets efficiently.
- Microsoft offers databases as a Platform-as-a-Service (PaaS). 
- Platform services are managed cloud services that provide the foundational building blocks for developing, deploying, and running applications without requiring users to manage the underlying infrastructure. 
- PaaS sits between Infrastructure-as-a-Service (IaaS) and Software-as-a-Service (SaaS) in the cloud service model.

### Model Layer
- The model layer involves the selection, training, and deployment of machine learning or AI models. 
- Models can be pretrained (for example: Azure OpenAI in Foundry Models) or custom-built using platforms like Azure Machine Learning. 
- This layer also includes tools for fine-tuning, evaluating, and versioning models to ensure they meet performance and accuracy requirements. 
- Microsoft Foundry, a unified Azure platform-as-a-service for enterprise AI operations, provides a comprehensive model catalog for application developers.

### Compute Layer
AI applications require compute resources to train and run models. Microsoft provides several platform options:

- Azure App Service for hosting web apps and APIs.
- Azure Functions for serverless, event-driven execution of AI tasks.
- Containers for scalable and portable deployment of AI models and services. 
- Azure Container Instances (ACI) offers lightweight, serverless container execution, perfect for AI workloads needing rapid deployment and simple scaling. 
- Azure Kubernetes Service (AKS) is a fully managed Kubernetes service that provides enterprise-level orchestration for AI workloads.

### Integration & Orchestration Layer
The integration and orchestration layer connects models and data with business logic and user interfaces. Foundry plays a key role here by offering:

- An agent Service for building intelligent agents that can reason and act.
- AI Tools like speech, vision, and language APIs.
- Software Development Kits (SDKs) and APIs for integrating AI capabilities into applications.
- Portal tools for managing models, agents, and workflows.

By using Foundry to build their applications, developers can embed intelligence directly within the data layer for smarter, more responsive applications. 

## Microsoft Foundry for AI
A unified, enterprise-grade platform for building, deploying, and managing AI applications and agents. 
- It consolidates models, agent orchestration, monitoring, and governance tools in one platform, offering production-grade infrastructure and security. 
- With Foundry, developers can seamlessly design, customize, and scale generative AI applications using a rich portal experience or integrated SDKs, without worrying about underlying infrastructure complexities.


Within Foundry's portal, you can work with:

- **Foundry Models**: Access to foundation and partner models (Azure OpenAI in Foundry Models, Anthropic, Cohere, etc.).
- **Agent Service**: Build and orchestrate multi-step AI workflows.
- **Foundry Tools**: Prebuilt Azure services (Vision, Language, Search, Document Intelligence).
- **Governance & Observability capabilities**: Centralized identity, policy, and monitoring for AI workloads.

### Foundry Models
- Supports thousands of models from Azure OpenAI (gpt-4, gpt-5), Anthropic, Cohere, Meta Llama, Mistral
- Provides model catalog, playground testing, deployment as agents, lifecycle management
- Offers region-specific deployment and version control

### Agent Service
- Builds production-ready AI agents that make decisions and automate workflows
- Handles orchestration, thread management, tool invocation, governance
- Supports low-code or code-first multi-agent systems
- Integrates with Azure Functions and Fabric

### Foundry Tools
- Comprehensive Azure services: speech, vision, language, document intelligence, content safety, embeddings
- Accessible via portal, APIs, or SDKs
- Over a dozen services for web/mobile applications
- Examples: Azure Vision (image analysis), Azure Language (text summarization, classification), Azure Speech (speech-to-text, text-to-speech)

### Governance and Observability
- Ensures responsible AI through compliance, identity management, risk mitigation
- Provides end-to-end visibility for performance, safety, operational efficiency
- Unified dashboard for metrics, lifecycle monitoring, feedback loops
- Embeds governance into AI development lifecycle for transparency and security

## Get started with Foundry
Once you create a Foundry project, you can access:

- The model catalog (foundation and partner models)
- Playgrounds for testing models
- Tools for deploying models, running evaluations, and creating agents
- A Management Center for user roles, quotas, and resource connections

### Azure Resource Organization and Access
Foundry projects are organized within Azure's resource hierarchy:
- **Tenant**: Azure Active Directory for identity and access management
- **Subscription**: Billing boundary and service access
- **Resource Group**: Logical container for Foundry project and related resources
- **Resources**: Foundry project, models, tools, and other Azure services

Access is managed through:
- Azure role-based access control (RBAC)
- Resource keys and endpoints for API authentication
- Management Center for user roles and quotas
- Integration with Azure networking and security


### Characteristics of Foundry offerings
Foundry models and tools are based on principles that dramatically improve speed-to-market:

- **Prebuilt and ready to use or customize**
- **Accessed through APIs**
- **Available on Azure**

#### Prebuilt and ready to use
- AI has been prohibitive for all but the largest technology companies because of several factors, including the large amounts of data required to train models, the massive amount of computing power needed, and the budget to hire specialist programmers. 
- Foundry makes AI accessible to businesses of all sizes by using pretrained machine learning models to deliver AI as a service. 
- Foundry uses high-performance Azure computing to deploy advanced AI models, making decades of research available to developers of all skill levels.

#### Accessed through APIs
- Foundry models and tools can be built into applications with APIs. 
- Secure communication with APIs is possible through authentication.
- Part of what an API does is to handle authentication. Whenever a request is made to use a Foundry resource, that request must be authenticated. 
- The endpoint describes how to reach the AI service resource instance that you want to use, in a similar way to the way a URL identifies a web site. When you view the endpoint for your resource, it looks something like:

```
https://cognitiveservices48.cognitiveservices.azure.com/
```

The resource key protects the privacy of your resource. To ensure your key is always secure, it can be changed periodically.

When you write code to access the resource, the keys and endpoint must be included in the authentication header. The authentication header sends an authorization key to the service to confirm that the application can use the resource.

## Understand Azure
icrosoft Azure is a cloud computing platform that provides a wide range of services to help individuals and organizations build, deploy, and manage applications through Microsoft-managed data centers. 
- It offers flexibility, scalability, and global reach. 
- Azure supports various programming languages, frameworks, and operating systems, allowing developers to work with the tools they prefer.

### Cloud capabilities
Azure delivers core cloud capabilities across four main areas
- Compute services include virtual machines, containers, and serverless functions that run workloads efficiently. 
- Storage services offer scalable and secure options for saving data, such as Blob Storage and Azure Files. 
- Networking services connect resources securely and reliably, using tools like Azure Virtual Network and Load Balancer. 
- Application services help developers build and host web apps, APIs, and mobile backends with ease.

### Understand how Azure organizes your resources
Azure organizes resources in a hierarchy:
- **Tenant**: Azure Active Directory instance for identity and access management
- **Subscription**: Billing boundary and access to Azure services
- **Resource Groups**: Logical containers for managing related resources together
- **Resources**: Individual services (VMs, databases, storage accounts)

Key benefits:
- Clear separation of concerns across departments/projects
- Simplified management by grouping related assets
- Easier policy application, usage monitoring, and deployment automation
- Essential for cloud governance and cost control

### Foundry runs on Azure
- Foundry is an AI development layer within Azure
- Uses Azure resource types and integrates with Azure networking, storage, security
- Foundry projects and hubs are Azure resources
- Managed like other Azure services (PaaS, IaaS, managed databases)
- Consistent framework for resource creation, deletion, availability, billing
- Start with Azure subscription → create Foundry project → develop AI application

### Summary



# Module 3: Introduction to Machine Learning Concepts (1200 XP)
### Introduction
**Machine learning** is a subset of AI that enables computers to learn and improve from experience without being explicitly programmed. It uses statistical techniques to enable machines to acquire knowledge from data and make predictions or decisions based on that knowledge.
### Machine learning models
**Machine learning models** are algorithms that can learn from data and make predictions or decisions based on that data.
1. Training data consists of past observations from features and their corresponding outcomes (labels). This observed data is used to teach the model.
2. An algorithm is applied to the data to identify patterns and relationships between the features (x) and the outcomes (y).
3. The result of the algorithm is a model that encapsulates the calculation dervied by the algorithm as a function. y=f(x)
4. After training, the model can be used for inferencing on new data to make predictions or decisions.
### Types of machine learning model
#### Supervised learning
training data consists of input features (x) and their corresponding correct outputs (y). Identify relationships between features and labels in past observations so that unknown labels can be predicted for new data.
- Regression: Label is predicted by model is numeric value
- Classification: Label is predicted by model is categorical value
#### Unsupervised learning
training data consists of input features (x) only. Identify patterns and relationships between the features (x) without knowing the correct outputs (y).
- Clustering: Group similar data points together based on their features.

### Regression
Regression models are trianed to predict numeric values based on input features. 
1. Split training data into training and validation sets.
2. Use algorithm to fit the training data to a model.
3. Use the validation data to test the model by predicting labels for the features
4. Compare the predicted labels with the actual labels to evaluate the model's performance.
#### Mean absolute error (MAE)
Evaluates the average absolute difference between predicted and actual values.
#### Mean squared error (MSE)
Evaluates the average squared difference between predicted and actual values. This is done to penalize larger errors more heavily.
#### Root mean squared error (RMSE)
Evaluates the square root of the average squared difference between predicted and actual values. This provides a measure of error in the same units as the target variable.
#### Coefficient of determination (R²)
Evaluates the proportion of the variance in the dependent variable that is predictable from the independent variable(s).
#### Iterative training
Training process that involves multiple cycles of learning and improvement with each cycle refining the model's parameters based on feedback from the data.
### Binary classification
Trained to predict binary outcomes (e.g., yes/no, true/false).
- We use an algorithm to fit the training data to a function that calculates probabilities for each class.
#### Confusion matrix 
A table used to describe the performance of a classification model on a set of test data for which the true values are known.
- True negative: The model correctly predicted the negative class. y^ = 0, y = 0
- True positive: The model correctly predicted the positive class. y^ = 1, y = 1
- False negative: The model incorrectly predicted the negative class. y^ = 0, y = 1
- False positive: The model incorrectly predicted the positive class. y^ = 1, y = 0

Correct predictions are on the diagonal, incorrect predictions are off the diagonal.
Accuracy = (True positive + True negative) / (Total predictions)
#### Recall
Proportion of positive cases that were correctly identified. 
- Recall = True positive / (True positive + False negative)
- How many actual positive cases were identified?
#### Precision
Proportion of predicted positive cases that were actually positive. 
- Precision = True positive / (True positive + False positive)
- How many predicted positive cases were actually positive?
#### F1 score
Harmonic mean of precision and recall. 
- F1 = 2 * (Precision * Recall) / (Precision + Recall)
#### Area under the ROC curve (AUC-ROC)
Measures the model's ability to distinguish between classes. Changes based on the threshold. 
- ROC (Receiver Operating Characteristic) curve: plots the true positive rate against the false positive rate at various threshold settings.
### Multiclass classification
Trained to predict categorical outcomes with more than two classes.
#### Training algorithm
##### One-vs-Rest (OvR)
- Train one binary classifier for each class against all other classes, each calcuilating the probability that the observation belongs to that class.
- The class with the highest probability is selected as the prediction.

### Clustering
Groups similar data points together based on their features. This doesn't require labeled data.
#### Training K-Means
1. feature (x) values are vectorized to define n-dimensional coordinate (for n features), we plot these coordinates on a graph
2. Decide how many clusters (k) we want to identify. The k points are randomly placed on the graph, these points are called centroids
3. Each data is assigned to its nearest centroid
4. The centroids are moved to the center of their assigned data points based mean distance
5. After the centeroid is moved, the data points may now be closer to a different centroid, so the data points are reassigned to clusters based on nearest centroid
6. The centeroid movement and cluster rellocation steps are repeated until the centroids no longer move significantly
#### Evaluation
No known label with  which to compare the clusters. Evaluation is based on how well the clusters are seperated from each other.
- **Average distance to cluster center**: How close on average each point in the cluster is to the centeroid of the cluster
- **Average distance to other cneter**: How close on abverge each point in the cluster is to the centeroid of all other clusters.
- **Maximum distance to cluster center**: Furthest distance between a point in the cluster to its centroid.
- **Silhouette score**: Value between -1 and 1 that summarizes ratio of ditstance between points in the same cluster and points in different clusers. Closer to 1 indicates better clustering.

### Deep learning
Emulate the way the human brain learns using artificial neural network that simulates electrochemical activity in biological neurons using mathemtical functions.
- Each neuron is a function that operates on inputs (x) and weight (w). The function is wrapped in an activation function that determines whether to pass the output on.
- Artificial neural networks are made up of multiple layers of neurons.
- Deep neural networks can be used for nlp and computer vision.
- When fitting the data to predict label y, the function f(x) isthe outer layer of the nested function in which each layer of the neural network encapsulates functions that operate on x and the weight w values associated with them
- The algorithm used to train the model invovles iteratively feeding feature values in the training data forward through the layers to calculate output values for y^, validating the model to evaluate how fasr off the predictions are from the actual values, and then adjusting the weights to minimize the error.
#### How neural networks learn
The weights in a neural network are central to how it calculates predicted values for labels. During the training process, the model learns the weights that will result in the most accurate predictions
1. Training and validation sets defined and training features are fed into the input layer
2. Neurons in each layer of the network apply their weights and feed the data through the network
3. The output layer produces a vector containing the calculated values.
4. The loss function compares the calculated values to the actual values to determine the error.
5. Since the entire network is one large nested function, optimzation fucntion can use differential calculus to evaluate tje influence of each weight in the network on the loss, and determine how they can be adjusted to reduce the loss. We can use gradient decent  in which each weight is increased or decreased to minimize loss.
6. The changes to the weight are backpropagated to the layers in the network, replacing the previously used values.
7. The process is repeated over multiple iterations (epochs) until the loss is minimize and the model predits acceptably accurately

### Summary

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

Features
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

# Module 5: Introduction to Generative AI and Agents (1000 XP)
## Introduction
## Large language models (LLMs)
LLMs encapsulate the linguistic and semantic relationships between the words and phrases in a vocabulary. The model can use these relationships to reason over natural language input and generate meaningful and relevant responses.

Fundamentally, LLMs are trained to generate completions based on prompts. Think of them as being super-powerful examples of the predictive text feature on many cellphones. A prompt starts a sequence of text predictions that results in a semantically correct completion. The trick is that the model understands the relationships between words and it can identify which words in the sequence so far are most likely to influence the next one; and use that to predict the most probable continuation of the sequence.

### Tokenization
The first step is to provide the model with a large vocabulary of words and phrases; and we do mean large. The latest generation of LLMs have vocabularies that consist of hundreds of thousands of tokens, based on large volumes of training data from across the Internet and other sources.

LLMs break down their vocabulary into tokens. Tokens include words, but also sub-words (like the "un" in "unbelievable" and "unlikely"), punctuation, and other commonly used sequences of characters. The first step in training a large language model therefore is to break down the training text into its distinct tokens, and assign a unique integer identifier to each one

### Transforming tokens with a transformer
- Now that we have a set of tokens with unique IDs, we need to find a way to relate them to one another. 
- To do this, we assign each token a vector (an array of multiple numeric values, like [1, 23, 45]). 
    - Each vector has multiple numeric elements or dimensions, and we can use these to encode linguistic and semantic attributes of the token to help provide a great deal of information about what the token means and how it relates to other tokens, in an efficient format.
- We need to transform the initial vector representations of the tokens into new vectors with linguistic and semantic characteristics embedded in them, based on the contexts in which they appear in the training data. Because the new vectors have semantic values embedded in them, we call them embeddings.

To accomplish this task, we use a transformer model. This kind of model consists of two "blocks":
- An encoder block that creates the embeddings by applying a technique called attention. 
    - The attention layer examines each token in turn, and determines how it's influenced by the tokens around it. To make the encoding process more efficient, multi-head attention is used to evaluate multiple elements of the token in parallel and assign weights that can be used to calculate the new vector element values. The results of the attention layer are fed into a fully connected neural network to find the best vector representation of the embedding.
- A decoder layer that uses the embeddings calculated by the encoder to determine the next most probable token in a sequence started by a prompt. The decoder also uses attention and a feed-forward neural network to make its predictions.

The key point is that the transformer helps capture linguistic and semantic characteristics of each token based on the contexts in which it's used. 

### Initial vectors and positional encoding
Initially, the token vector values are assigned randomly, before being fed through the transformer to create embedding vectors. The token vectors are fed into the transformer along with a positional encoding that indicates where the token appears in the sequence of training text (we need to do this because the order in which tokens appear in the sequence is relevant to how they relate to one another).

### Attention and embeddings
To determine the vector representations of tokens that include embedded contextual information, the transformer uses attention layers. 
- **An attention layer** considers each token in turn, within the context of the sequence of tokens in which it appears. The tokens around the current one are weighted to reflect their influence and the weights are used to calculate the element values for the current token's embedding vector. 
    - For example, when considering the token "bark" in the context of "I heard a dog bark", the tokens for "heard" and "dog" will be assigned more weight than "I" or "a", since they're stronger indicators for "bark".

- Initially, the model doesn't "know" which tokens influence others; but as it's exposed to larger volumes of text, it can iteratively learn which tokens commonly appear together, and start to find patterns that help assign values to the vector elements that reflect the linguistic and semantic characteristics of the tokens, based on their proximity and frequency of use together. 
    - The process is made more efficient by using multi-head attention to consider different elements of the vectors in parallel.

- Embeddings represent a token within a particular context; and some tokens might be used to mean multiple things. 
    - For example, the bark of a dog is different from the bark of a tree! Tokens that are commonly used in multiple contexts can produce multiple embeddings.

- Because the dimensions are calculated based on how the tokens relate linguistically to one another, tokens that are used in similar contexts (and therefore have similar meanings) result in vectors with similar directions. 
    - For example, the embeddings for "dog" and "puppy" point in more or less the same direction, which isn't too different from the embedding for "cat"; but very different from the embedding for "skateboard" or "car". We can measure how close tokens are to one another semantically by calculating the cosine similarity of their vectors.

### Predicting completions from prompts
Now that we have a set of embeddings that encapsulate the contextual relationship between tokens, we can use the decoder block of a transformer to iteratively predict the next word in a sequence based on a starting prompt.

Once again, **attention** is used to consider each token in context; but this time the context to be considered can only include the tokens that precede the token we're trying to predict. 
- The decoder model is trained, using data for which we already have the full sequence, by applying a technique called **masked attention**; in which the tokens after the current token are ignored. 
- Since we already know the next token during training, the transformer can compare it to the predicted token and adjust the learned weights in later training iterations to reduce the error in the model.

When predicting a new completion, for which the next tokens are unknown, the attention layers calculate possible vectors for the next token and the feed-forward network is used to help determine the most probable candidate. The predicted value is then added to the sequence, and the whole process repeats to predict the next token; and so on, until the decoder predicts that the sequence has ended.
- For example, given the sequence "When my dog was a ...", the model will evaluate the tokens in the sequence so far, use attention to assign weights, and predict that the next most probable token is "puppy" rather than, say, "cat" or "skateboard".
## Prompts
## AI agents
## Summary

# Module 6: Get Started with Generative AI and Agents in Microsoft Foundry (800 XP)
## Introduction
## Understand generative AI applications
## Understand generative AI development in Foundry
## Understand Foundry's model catalog
## Understand Foundry capabilities
## Understand observability
## Summary

# Module 7: Introduction to Text Analysis Concepts (1000 XP)
## Introduction
## Tokenization
## Statistical text analysis
## Semantic language models
## Summary

# Module 8: Get Started with Text Analysis in Microsoft Foundry (800 XP)
## Introduction
## Understand natural language processing on Azure
## Understand Azure Language's text analysis capabilities
## Azure Language's conversational AI capabilities
## Azure Translator capabilities
## Get started in Microsoft Foundry
## Summary

# Module 9: Introduction to AI Speech Concepts (800 XP)
## Introduction
## Speech-enabled solutions
## Speech recognition
## Speech synthesis
## Summary

# Module 10: Get Started with Speech in Microsoft Foundry (1000 XP)
## Introduction
## Understand speech recognition and synthesis
## Get started with speech on Azure
## Use Azure Speech
## Summary

# Module 11: Introduction to Computer Vision Concepts (900 XP)
## Introduction
## Computer vision tasks and techniques
## Images and image processing
## Convolutional neural networks
## Vision transformers and multimodal models
## Image generation
## Summary

# Module 12: Get Started with Computer Vision in Microsoft Foundry (800 XP)
## Introduction
## Understand Foundry Tools for computer vision
## Understand Azure Vision Image Analysis capabilities
## Understand Azure Vision's Face service capabilities
## Get started in Microsoft Foundry portal
## Summary

# Module 13: Introduction to AI-Powered Information Extraction Concepts (1000 XP)
## Introduction
## Overview of information extraction
## Optical character recognition (OCR)
## Field extraction and mapping
## Summary

# Module 14: Get Started with AI-Powered Information Extraction in Microsoft Foundry (1000 XP)
## Introduction
## Azure AI services for information extraction
## Extract information with Azure Vision
## Extract multimodal information with Azure Content Understanding
## Extract information from forms with Azure Document Intelligence
## Create a knowledge mining solution with Azure AI Search
## Summary
