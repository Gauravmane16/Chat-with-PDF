# PDF Chat Tool 📄

A powerful interactive tool built with Streamlit that allows users to chat with their PDF documents using OpenAI's GPT-3.5 model. This application enables users to upload multiple PDF files and ask questions about their content, receiving accurate responses based on the document context.

Visit https://github.com/Gauravmane16/Chat-with-PDF for source code.

## 🌟 Features

- **Multiple PDF Support**: Upload and process multiple PDF files simultaneously
- **Interactive Chat Interface**: Natural conversation-style interaction with your documents
- **Context-Aware Responses**: Get accurate answers based on the content of your PDFs
- **Persistent Chat History**: View and track your conversation history
- **Secure API Key Management**: Safe handling of OpenAI API keys
- **User-Friendly Interface**: Clean and intuitive design with clear instructions
- **Error Handling**: Comprehensive error messages and user guidance

## 🔧 Prerequisites

Before running the application, make sure you have the following:

- Python 3.7+
- OpenAI API key ([Get one here](https://platform.openai.com/api-keys))
- Required Python packages (listed in requirements.txt)

## 📦 Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd pdf-chat-tool
```

2. Create and activate a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows, use: venv\Scripts\activate
```

3. Install the required packages:
```bash
pip install -r requirements.txt
```

## 📝 Required Packages

Create a `requirements.txt` file with the following dependencies:
```
streamlit
PyPDF2
langchain
langchain-openai
faiss-cpu
spacy
```

## 🚀 Usage

1. Start the application:
```bash
streamlit run app.py
```

2. Open your web browser and navigate to the provided local URL (typically `http://localhost:8501`)

3. In the sidebar:
   - Enter your OpenAI API key
   - Upload your PDF file(s)
   - Click the "Embed" button to process the documents

4. Start asking questions about your PDF content in the chat interface

## 💡 How It Works

1. **PDF Processing**:
   - The tool reads and extracts text from uploaded PDFs
   - Text is split into manageable chunks
   - Chunks are embedded using OpenAI's embedding model
   - Embeddings are stored in a FAISS vector database

2. **Query Processing**:
   - User questions are processed using OpenAI's GPT-3.5 model
   - The system retrieves relevant context from the vector database
   - Responses are generated based on the document context
   - Conversations are maintained in the session state

## 🔐 Security

- API keys are stored securely in session state
- API keys are never displayed after entry
- PDF content is processed locally
- No data is permanently stored on any external servers

## ⚠️ Important Notes

- Keep your OpenAI API key secure and never share it publicly
- Large PDF files may take longer to process
- The quality of responses depends on the clarity and content of your PDFs
- Make sure your API key has sufficient credits for your usage

## 🛠️ Troubleshooting

Common issues and solutions:

1. **PDF Processing Error**:
   - Ensure PDF files are not corrupted
   - Check if PDFs are password protected
   - Try with smaller PDF files first

2. **API Key Issues**:
   - Verify the API key is entered correctly
   - Ensure the API key has not expired
   - Check if you have sufficient API credits

3. **Response Issues**:
   - Make sure PDFs are properly embedded before asking questions
   - Try rephrasing your question if answers are not relevant
   - Check if the information exists in your uploaded PDFs

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Built with [Streamlit](https://streamlit.io/)
- Powered by [OpenAI](https://openai.com/)
- Uses [LangChain](https://python.langchain.com/) for document processing
- Vector search powered by [FAISS](https://github.com/facebookresearch/faiss)

## 📞 Support

If you encounter any issues or have questions, please:
1. Check the troubleshooting section
2. Review existing GitHub issues
3. Create a new issue with a detailed description of your problem

---

Made with ❤️ for the developer community