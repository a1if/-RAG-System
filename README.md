# RAG-System
The project is a Retrieval-Augmented Generation (RAG) application that uses advanced models from Google and Groq to provide accurate, context-aware, and multilingual responses, enabling intelligent assistants to reason beyond simple fact retrieval.

## Setup

To get started with this project, follow these steps:

### 1. Clone the Repository

```bash
git clone https://github.com/a1if/-RAG-System.git
cd -RAG-System
```

### 2. Install Dependencies

This project uses `pip` for dependency management. Ensure you have Python 3.8+ installed.

```bash
pip install -r requirements.txt
```




### 3. Running the Application

This project uses `start_services.py` to launch both the backend (FastAPI) and frontend (Streamlit) services. This script automatically finds available ports and starts the services.

```bash
python start_services.py
```

Upon successful execution, the script will output the URLs for the backend API and the frontend application. Open the frontend URL in your web browser to interact with the RAG system.




## Project Structure

The repository is organized into the following key directories:

- `app/`: Contains the backend FastAPI application, including API endpoints, configuration, dependencies, and core logic for document loading, pipeline processing, and web search integration.
- `frontend/`: Houses the Streamlit-based user interface for interacting with the RAG system.
- `vectorstore/`:  contains components related to the vector database used for retrieval, such as vector store initialization and management.





### 2.1. Create a `.env` file

Before running the project, create a `.env` file in the root directory of the project. This file will store your API keys and other sensitive information. A typical `.env` file might look like this:

```
GOOGLE_API_KEY=your_google_api_key_here
GROQ_API_KEY=your_groq_api_key_here
GEMINI_API_KEY=
BACKEND_HOST=0.0.0.0
BACKEND_PORT=8000
TAVILY_API_KEY=
GROQ_API_KEY=
```

Replace `your_google_api_key_here` , `your_groq_api_key_here` and 'TAVILY_API_KEY' with your actual API keys. You can obtain these from the respective platform websites.
