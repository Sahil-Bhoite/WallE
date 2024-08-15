# WallE: Walmart's AI Shopping Assistant

## Overview

WallE is an AI-powered shopping assistant developed for the Walmart Hackathon 2024. It aims to revolutionize the retail experience by providing customers with an intelligent, conversational interface to explore and learn about Walmart's product offerings.

## Features

- **AI-Powered Conversations**: Utilizes advanced language models to understand and respond to customer queries.
- **Product Knowledge Base**: Integrates with Walmart's product catalog to provide accurate and up-to-date information.
- **User-Friendly Interface**: Built with Streamlit for a smooth and interactive user experience.
- **Efficient Data Processing**: Employs multiprocessing for handling large datasets quickly.

## Technologies Used

- Python
- Streamlit
- LangChain
- FAISS (Facebook AI Similarity Search)
- Google Palm Embeddings
- Ollama (for offline model deployment)

## Setup and Installation

1. Clone the repository:
   ```
   git clone https://github.com/your-username/WallE.git
   cd WallE
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Set up your Google API Key:
   - Create a `config.py` file in the root directory.
   - Add your Google API Key:
     ```python
     GOOGLE_API_KEY = "your-api-key-here"
     ```

4. Prepare your product data:
   - Ensure you have an `Apple.csv` file with the product information in the root directory.

5. Set up Ollama:
   - Follow the instructions at [Ollama's official website](https://ollama.ai/) to install and set up Ollama on your system.
   - Pull the required model (e.g., "walle") using Ollama.

## Usage

To run the WallE assistant:

```
streamlit run Walle.py
```

Navigate to the provided local URL in your web browser to interact with the AI shopping assistant.

## Project Structure

- `Walle.py`: Main application file containing the Streamlit interface and core logic.
- `config.py`: Configuration file for API keys and other settings.
- `Apple.csv`: Sample product data file.
- `README.md`: This file, containing project documentation.

## Contributing

Contributions to WallE are welcome! Please feel free to submit pull requests, create issues or spread the word.

## License

License - see the [LICENSE](https://github.com/Sahil-Bhoite/WallE/blob/main/LICENSE) file for details.

## Author

Developed by Sahil Bhoite
- LinkedIn: [https://www.linkedin.com/in/sahil-bhoite/](https://www.linkedin.com/in/sahil-bhoite/)
- GitHub: [https://github.com/Sahil-Bhoite](https://github.com/Sahil-Bhoite)

## Acknowledgements

- Walmart for organizing the Hackathon 2024
- The open-source community for the amazing tools and libraries used in this project

---



