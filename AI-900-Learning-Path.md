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

## Module 1: Overview of AI Concepts (1100 XP)
### Generative AI and agents
**Key Concepts:**
- **Generative AI**: Branch of AI that enables software applications to generate new content
- **Language Models**: Trained with huge volumes of data - often documents from the Internet or other public sources of information
- **Prompts**: Natural language statements or questions that users interact with to initiate generation of meaningful responses
- **Semantic Relationships**: Models "know" how words relate to one another, enabling generation of meaningful text sequences
- **Large Language Models (LLMs)**: Powerful, generalize well, but more costly to train and use
- **Small Language Models (SLMs)**: Work well for specific topic areas, easily deployed for local applications and device agents

### Text and natural language
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

### Speech
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

### Computer vision
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

### Information extraction
**Key Concepts:**
- **Information Extraction**: Finding information and unlocking insights in unstructured data sources, such as scanned documents and forms, images, and audio or video recordings
- **Optical Character Recognition (OCR)**: Computer vision technology that can identify the location of text in an image
- **Field Extraction**: OCR is often combined with an analytical model that can interpret individual values in the document, and so extract specific fields
- **Advanced Models**: More advanced models that can extract information from audio recording, images, and videos are becoming more readily available

**Data and Insight Extraction Scenarios:**
- Automated processing of forms and other documents in a business process - for example, processing an expense claim
- Large-scale digitization of data from paper forms. For example, scanning and archiving census records
- Identifying key points and follow-up actions from meeting transcripts or recordings

### Responsible AI
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

### Summary
**Module 1 Key Takeaways:**
- **AI Fundamentals**: AI encompasses multiple technologies that simulate human intelligence, including generative AI that creates new content and traditional AI that analyzes existing data
- **Core AI Capabilities**: Text and natural language processing, speech recognition and synthesis, computer vision, and information extraction enable machines to understand and interact with the world
- **Generative AI**: Uses language models trained on vast data to generate content through prompts, with LLMs offering broad capabilities and SLMs providing focused, efficient solutions
- **Responsible AI**: Essential principles of fairness, reliability, privacy, inclusiveness, transparency, and accountability ensure ethical and trustworthy AI development and deployment

## Module 2: Get Started in Microsoft Foundry (1000 XP)
### Introduction
### What is an AI application?
### Components of an AI application
### Microsoft Foundry for AI
### Get started with Foundry
### Understand Azure
### Summary

## Module 3: Introduction to Machine Learning Concepts (1200 XP)
### Introduction
### Machine learning models
### Types of machine learning model
### Regression
### Binary classification
### Multiclass classification
### Clustering
### Deep learning
### Summary

## Module 4: Get Started with Machine Learning in Azure (1000 XP)
### Introduction
### Define the problem
### Get and prepare data
### Train the model
### Use Azure Machine Learning studio
### Integrate a model
### Summary

## Module 5: Introduction to Generative AI and Agents (1000 XP)
### Introduction
### Large language models (LLMs)
### Prompts
### AI agents
### Summary

## Module 6: Get Started with Generative AI and Agents in Microsoft Foundry (800 XP)
### Introduction
### Understand generative AI applications
### Understand generative AI development in Foundry
### Understand Foundry's model catalog
### Understand Foundry capabilities
### Understand observability
### Summary

## Module 7: Introduction to Text Analysis Concepts (1000 XP)
### Introduction
### Tokenization
### Statistical text analysis
### Semantic language models
### Summary

## Module 8: Get Started with Text Analysis in Microsoft Foundry (800 XP)
### Introduction
### Understand natural language processing on Azure
### Understand Azure Language's text analysis capabilities
### Azure Language's conversational AI capabilities
### Azure Translator capabilities
### Get started in Microsoft Foundry
### Summary

## Module 9: Introduction to AI Speech Concepts (800 XP)
### Introduction
### Speech-enabled solutions
### Speech recognition
### Speech synthesis
### Summary

## Module 10: Get Started with Speech in Microsoft Foundry (1000 XP)
### Introduction
### Understand speech recognition and synthesis
### Get started with speech on Azure
### Use Azure Speech
### Summary

## Module 11: Introduction to Computer Vision Concepts (900 XP)
### Introduction
### Computer vision tasks and techniques
### Images and image processing
### Convolutional neural networks
### Vision transformers and multimodal models
### Image generation
### Summary

## Module 12: Get Started with Computer Vision in Microsoft Foundry (800 XP)
### Introduction
### Understand Foundry Tools for computer vision
### Understand Azure Vision Image Analysis capabilities
### Understand Azure Vision's Face service capabilities
### Get started in Microsoft Foundry portal
### Summary

## Module 13: Introduction to AI-Powered Information Extraction Concepts (1000 XP)
### Introduction
### Overview of information extraction
### Optical character recognition (OCR)
### Field extraction and mapping
### Summary

## Module 14: Get Started with AI-Powered Information Extraction in Microsoft Foundry (1000 XP)
### Introduction
### Azure AI services for information extraction
### Extract information with Azure Vision
### Extract multimodal information with Azure Content Understanding
### Extract information from forms with Azure Document Intelligence
### Create a knowledge mining solution with Azure AI Search
### Summary
