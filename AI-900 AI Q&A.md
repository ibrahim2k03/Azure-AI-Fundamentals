# Azure AI Fundamentals - Exam Q&A

## What is an AI application?

**Q1: What is Artificial Intelligence?**
- Systems designed to perform tasks that typically require human intelligence (reasoning, problem-solving, perception, language understanding).

**Q2: What is Responsible AI?**
- Emphasizes fairness, transparency, and ethical use of AI technologies.

**Q3: What are the key AI workloads and their primary functions?**
- Generative AI: Create
- Agents and automation: Automate
- Speech: Listen
- Text analysis: Read
- Computer Vision: See
- Information Extraction: Extract

**Q4: What is Machine Learning?**
- Enables machines to learn patterns from data and improve performance without explicit programming.

**Q5: What is the difference between AI and ML?**
- AI is the broad goal of creating intelligent systems. ML is the primary method to achieve AI using data-driven algorithms.

## Types of Machine Learning

**Q6: What are the main types of Machine Learning?**
- Supervised: Regression, classification
- Unsupervised: Clustering
- Deep Learning: Neural networks for complex tasks
- Generative AI: Creates new content (text, images, audio, code)

## AI Applications

**Q7: What are the two key characteristics of AI applications?**
- Model-powered: Use trained models to process inputs and generate outputs
- Dynamic: Improve over time through retraining or fine-tuning

**Q8: What are the typical ways people interact with AI applications?**
- Conversational Interfaces: Chatbots or voice assistants
- Embedded Features: AI integrated into apps (autocomplete, image recognition)
- Decision Support: Insights or predictions for informed choices
- Automation: Handle repetitive tasks

## Components of an AI Application

**Q9: What are the four layers of an AI application?**
- Data layer, Model layer, Compute layer, Integration & Orchestration layer

**Q10: What is the primary role of the Data Layer?**
- Provide foundation for collecting, storing, and managing data for training, inference, and decision-making.

**Q11: What is the primary role of the Model Layer?**
- Select, train, deploy, and manage ML models that process data and generate AI-driven insights.

**Q12: What is the primary role of the Compute Layer?**
- Provide computational resources and infrastructure to train, deploy, and run AI models efficiently.

**Q13: What is the primary role of the Integration & Orchestration Layer?**
- Connect AI models and data with business logic and user interfaces to create intelligent workflows.

**Q14: What are common data sources in the Data Layer?**
- Structured databases (Azure SQL, PostgreSQL), unstructured data (documents, images), real-time streams.

**Q15: What is Platform-as-a-Service (PaaS)?**
- Managed cloud services providing foundational building blocks without managing underlying infrastructure.

**Q16: What happens in the Model Layer?**
- Selection, training, deployment of ML models. Pretrained (Azure OpenAI) or custom-built (Azure ML).

**Q17: What compute options does Microsoft provide for AI applications?**
- Azure App Service: Web apps and APIs
- Azure Functions: Serverless execution
- Containers: Scalable deployment (ACI, AKS)

**Q18: What does the Integration & Orchestration Layer provide?**
- Agent Service: Build intelligent agents
- AI Tools: Speech, vision, language APIs
- SDKs and APIs: Integration capabilities
- Portal tools: Management interfaces

## Microsoft Foundry for AI

**Q19: What is Microsoft Foundry?**
- Unified enterprise-grade platform for building, deploying, and managing AI applications and agents.

**Q20: What can you access within Foundry's portal?**
- Foundry Models: Foundation and partner models
- Agent Service: Multi-step AI workflows
- Foundry Tools: Prebuilt Azure services
- Governance & Observability: Centralized monitoring

**Q21: What models does Foundry support?**
- Thousands of models from Azure OpenAI, Anthropic, Cohere, Meta Llama, Mistral. Provides catalog, playground, deployment, lifecycle management.

**Q22: What does the Agent Service do?**
- Builds production-ready AI agents that make decisions and automate workflows. Supports low-code or code-first systems.

**Q23: What are Foundry Tools?**
- Azure services: speech, vision, language, document intelligence, content safety, embeddings. Accessible via portal, APIs, SDKs.

**Q24: What do Governance and Observability provide?**
- Responsible AI through compliance, identity management, risk mitigation. End-to-end visibility with unified dashboard.

## Get Started with Foundry

**Q25: What can you access once you create a Foundry project?**
- Model catalog
- Playgrounds for testing
- Deployment and evaluation tools
- Management Center

**Q26: How are Foundry projects organized within Azure's resource hierarchy?**
- Tenant: Azure AD for identity
- Subscription: Billing boundary
- Resource Group: Logical container
- Resources: Individual services

**Q27: How is access managed in Foundry?**
- Azure RBAC
- Resource keys and endpoints
- Management Center
- Azure networking and security integration

**Q28: What are the characteristics of Foundry offerings?**
- Prebuilt and ready to use
- Accessed through APIs
- Available on Azure

**Q29: How does Foundry make AI accessible?**
- Uses pretrained models to deliver AI as a service, making research available to all skill levels.

**Q30: How do APIs work with Foundry resources?**
- Models and tools built into applications via APIs using resource keys (authentication) and endpoints (access).

## Understand Azure

**Q31: What is Microsoft Azure?**
- Cloud computing platform providing services to build, deploy, and manage applications through Microsoft data centers.

**Q32: What are the four main areas of Azure cloud capabilities?**
- Compute: VMs, containers, serverless functions
- Storage: Blob Storage, Azure Files
- Networking: Virtual Network, Load Balancer
- Application services: Web apps, APIs, mobile backends

**Q33: How does Azure organize resources?**
- Tenant: Azure AD instance
- Subscription: Billing boundary
- Resource Groups: Logical containers
- Resources: Individual services

**Q34: What are the key benefits of Azure's resource organization?**
- Clear separation of concerns
- Simplified management
- Easier policy application and monitoring
- Essential for governance and cost control

**Q35: How does Foundry run on Azure?**
- AI development layer using Azure resource types, integrating with Azure networking, storage, and security.

**Q36: What is the development workflow starting from Azure?**
- Azure subscription → Foundry project → AI application development
