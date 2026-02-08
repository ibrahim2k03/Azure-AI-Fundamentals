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


