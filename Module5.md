# Module 5: Introduction to Generative AI and Agents (1000 XP)
## Introduction
## Large language models (LLMs)
LLMs encapsulate the linguistic and semantic relationships between the words and phrases in a vocabulary. The model can use these relationships to reason over natural language input and generate meaningful and relevant responses.

Fundamentally, LLMs are trained to generate completions based on prompts. Think of them as being super-powerful examples of the predictive text feature on many cellphones. A prompt starts a sequence of text predictions that results in a semantically correct completion. The trick is that the model understands the relationships between words and it can identify which words in the sequence so far are most likely to influence the next one; and use that to predict the most probable continuation of the sequence.

### Tokenization
Tokenization is the process where LLMs break down text into smaller units called tokens, which can be words, sub-words, or characters. The first step in training a large language model therefore is to break down the training text into its distinct tokens, and assign a unique integer identifier to each one

### Transforming tokens with a transformer
- Now that we have a set of tokens with unique IDs, we need to find a way to relate them to one another. 
- To do this, we assign each token a vector (an array of multiple numeric values, like [1, 23, 45]). 
    - Each vector has multiple numeric elements or dimensions, and we can use these to encode linguistic and semantic attributes of the token to help provide a great deal of information about what the token means and how it relates to other tokens, in an efficient format.
- We need to transform the initial vector representations of the tokens into new vectors with linguistic and semantic characteristics embedded in them, based on the contexts in which they appear in the training data. Because the new vectors have semantic values embedded in them, we call them embeddings.

**Embeddings** are vector based numeric representations of tokens that capture their meaning and relationships to other tokens in the context of the training data.

To accomplish this task, we use a transformer model. This kind of model consists of two "blocks":
- An encoder block that creates the embeddings by applying a technique called attention. 
    - The **attention layer** examines each token in turn, and determines how it's influenced by the tokens around it.
        - To make the encoding process more efficient, multi-head attention is used to evaluate multiple elements of the token in parallel and assign weights that can be used to calculate the new vector element values. 
        - The results of the attention layer are fed into a fully connected neural network to find the best vector representation of the embedding.
- A decoder layer that uses the embeddings calculated by the encoder to determine the next most probable token in a sequence started by a prompt. 
    - The decoder also uses attention and a feed-forward neural network to make its predictions.

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
The input you give to an LLM to get a response. 
    - **System prompts** that set the behavior and tone of the model, and any constraints it should adhere to. 
    - **User prompts** that evoke a response to a specific question or instruction. 
        - User prompts can be entered by a human user in a chat application; or in some cases generated by the application on the user’s behalf. 
        -The model responds to user prompts while obeying the overall guidance in the system prompt.

Conversation history
To keep a conversation consistent and relevant, generative AI apps often keep track of the conversation history.

Key considerations for adopting Generative AI include
- Establish Clear Governance and Responsible AI Policies
- Prioritize High-Value Use Cases Across Corporate Functions
- Mitigate Risks Around Privacy, Security, and Compliance
- Foster Organizational Readiness and Cultural Adaptation
- Measure Impact and Build Accountability
- Stay Ahead of Strategic and Competitive Shifts

Retrieval augmented generation (RAG)
To add even more context, this approach involves retrieving information, like documents or emails, and using it to augment the prompt with relevant data. 


Diagram of a retrieval augmented generation being used to provide context in a prompt.

## AI agents
Agents are software applications built on generative AI that can reason over and generate natural language, automate tasks by using tools, and respond to contextual conditions to take appropriate action.

AI agents have three key elements:
- **A large language model**: This is the agent's brain; using generative AI for language understanding and reasoning.
- **Instructions**: A system prompt that defines the agent’s role and behavior. Think of it as the agent’s job description.
- **Tools**: These are what the agent uses to interact with the world. Tools can include:
    - Knowledge tools that provide access to information, like search engines or databases.
    - Action tools that enable the agent to perform tasks, such as sending emails, updating calendars, or controlling devices.


Multi-agent systems
Multiple agents can collaborate—each with its own specialty. One might gather data, another might analyze it, and a third might take action. Together, they form an AI-powered workforce that can handle complex workflows, just like a human team.

