
# AskWOLPERT

_Interact with a talking book “Jinnah of Pakistan” by Stanley Wolpert through this GenAI chatbot !!!_

This repository contains the code for AskWOLPERT, a GenAI chatbot, designed as a GenAI project that can be a starting point and an indicator of something really useful.
The idea is that I used Langchain framework and Pinecone vectorstore to create a RAG-powered LLM that answers the questions related to Indo-Pak history, especially to the personality of Muhammad Ali Jinnah. It uses the authentic books as a context (currently on one book but can easily be scaled to include a whole library). I have used Meta's Llama 2 (an opensource LLM) with the of Ollama to build this RAG chatbot.


## Features

- Talk to any book or library of books
- Chat with authentic and selected books 
- Quick access to the desired information
- Summarized/ Concise information


## Tech Stack

- [LangChain](https://www.langchain.com/)
- [Pinecone](https://www.pinecone.io/)
- [Ollama](https://ollama.com/)


## Deployment

To deploy this project run, clone the repository, install Ollama and download Llama2, add Pinecone API key and run following command in cmd

```bash
  python app.py
```

