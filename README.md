# Platform Chatbot

The Platform Chatbot is a conversational AI assistant that helps users with various tasks and information related to the platform. It can provide explanations on game rules, answer questions, and guide users through common platform functionalities.

## 🛠️ Tech Stack  

<p align="left">
  <img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/python/python-original.svg" width="50" height="50"/>
  <img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/fastapi/fastapi-original.svg" width="50" height="50"/>
  <img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/docker/docker-original.svg" width="50" height="50"/>
  <img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/azure/azure-original.svg" width="50" height="50"/>
  <img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/gitlab/gitlab-original.svg" width="50" height="50"/>
</p> 

- **Language:** Python  
- **Framework:** FastAPI
- **Vector Search:** HuggingFace Sentence Transformers  
- **Containerization:** Docker  
- **Cloud:** Azure App Service  
- **CI/CD:** GitLab Pipelines 

## Features

- **Game Rules Explanations**: The chatbot can explain the rules for a variety of games, including card games, board games, word games, and more.
- **General Question Answering**: Users can ask the chatbot about various topics, and it will provide helpful information to the best of its abilities.
- **Platform Guidance**: The chatbot can guide users through using the platform's features, such as the registration process, account settings, and more.

## Getting Started

### Installation

To set up the Platform Chatbot locally, follow these steps:

1. Clone the repository:
   ```
   git clone https://github.com/HopeyCodeDS/RAG-based-chatbot-integration
   ```
2. Navigate to the project directory:
   ```
   cd RAG-based-chatbot-integration
   ```
3. Create a virtual environment and activate it:
   ```
   python -m venv .venv
   source .venv/bin/activate
   ```
4. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```
5. Set the necessary environment variables:

6. Start the application:
   ```
   python app/main.py
   ```
7. The chatbot should now be running on `http://localhost:8000`.

### Project Structure

The Platform Chatbot project has the following structure:

```
chatbot/
├── app/
│   ├── main.py
│   └── models.py
├── data/
│   ├── game_rules/
│   └── platform_docs/
├── get_embedding_function.py
├── query_data.py
├── populate_database.py
├── Dockerfile
├── requirements.txt
└── .gitlab-ci.yml
```

- `app/main.py`: The main entry point of the application, handling API requests and responses.
- `app/models.py`: Defines the data models used in the application.
- `data/`: Contains the data files used by the application, including game rules and platform documentation.
- `get_embedding_function.py`: Defines the function for generating text embeddings using the HuggingFace Sentence Transformers.
- `query_data.py`: Handles the logic for querying and retrieving information, such as game rules.
- `populate_database.py`: 
- `Dockerfile`: Defines the Docker image for the application.
- `requirements.txt`: Lists the Python dependencies required for the project.
- `.gitlab-ci.yml`: Defines the GitLab CI/CD pipeline for building and deploying the application.

## 💻 Usage

Once running, access the chatbot via the platform UI.
Simply type your query (e.g., “What are the rules of chess?”), and the chatbot responds.

## ☁️ Deployment

Hosted on Azure App Service via Docker container

CI/CD with GitLab:

On commit → build Docker image → push → deploy automatically
