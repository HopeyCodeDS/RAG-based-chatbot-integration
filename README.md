# Platform Chatbot

The Platform Chatbot is a conversational AI assistant that helps users with various tasks and information related to the platform. It can provide explanations on game rules, answer questions, and guide users through common platform functionalities.

## Features

- **Game Rules Explanations**: The chatbot can explain the rules for a variety of games, including card games, board games, word games, and more.
- **General Question Answering**: Users can ask the chatbot about various topics, and it will provide helpful information to the best of its abilities.
- **Platform Guidance**: The chatbot can guide users through using the platform's features, such as the registration process, account settings, and more.

## Getting Started

### Installation

To set up the Platform Chatbot locally, follow these steps:

1. Clone the repository:
   ```
   git clone https://gitlab.com/kdg-ti/integration-5/2024-2025/team7/chatbot.git
   ```
2. Navigate to the project directory:
   ```
   cd chatbot
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

## Usage

To use the Platform Chatbot, simply navigate to the platform's website and look for the chatbot interface. You can start a conversation by typing your message, and the chatbot will respond with the relevant information.

## Deployment

The Platform Chatbot is deployed on the Azure App Service, using a Docker container. The deployment process is automated through a GitLab CI/CD pipeline, which handles the building and deploying of the Docker image.

## Contributing

If you would like to contribute to the development of the Platform Chatbot, please follow these steps:

1. Fork the repository.
2. Create a new branch for your feature or bug fix.
3. Make your changes and test them thoroughly.
4. Submit a pull request, providing a detailed description of the changes you've made.

## Contact

Team 7 - Integration 5
