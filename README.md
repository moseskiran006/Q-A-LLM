# Q-A-LLM

# Legal Document Question-Answering System

## 📄 Project Overview
A robust, AI-powered PDF question-answering tool designed for legal document analysis, utilizing Google's Gemini AI and advanced text processing techniques.

## 🏗️ System Architecture

### Key Components
1. **PDF Text Extraction**
   - Uses PyPDF2 for document parsing
   - Implements advanced text cleaning and normalization
   - Handles multi-page document processing

2. **Text Preprocessing**
   - Advanced text chunking algorithm
   - Semantic segmentation of document content
   - Removes noise and irrelevant formatting

3. **Retrieval Mechanism**
   - TF-IDF vectorization
   - Cosine similarity-based context matching
   - Top-k relevant chunk selection

4. **Question Answering**
   - Gemini Pro AI-powered response generation
   - Contextual prompt engineering
   - Confidence scoring system

### Architectural Diagram
```
[PDF Input] 
    ↓ 
[Text Extraction]
    ↓
[Text Preprocessing & Chunking]
    ↓
[Semantic Search & Retrieval]
    ↓
[Gemini AI Question Answering]
    ↓
[Response with Confidence Score]
```

## 🛠️ Prerequisites

### System Requirements
- Python 3.8+
- Google Cloud Account
- Gemini API Access

### Required Tools
- `pip` package manager
- Virtual environment (recommended)

## 🚀 Installation Steps

1. Clone the repository
``` bash
git clone <repo>
```

2. Install Dependencies
```bash
pip install -r requirements.txt
```

3. Configure Google Gemini API
- Create a Google Cloud account
- Enable Gemini API
- Generate an API key

4. On command line use 
```bash
setx GEMINI_API_KEY  "YOUR-API-KEY" 
```
``` bash
$env:GEMINI_API_KEY = "YOUR-API-KEY"
``` 
```bash

echo  %YOUR-API-KEY%
```

## 🖥️ Usage

### Command Line Interface
```bash
python3 app.py /path/to/legal/document.pdf "Your specific question about the document"
```

### Example
```bash
python3 question-answering.py contract.pdf "What are the termination conditions?"
```

## ⚠️ Major Pitfalls of Current Design

1. **Context Limitation**
   - Restricted by AI model's context window
   - May miss nuanced details in very long documents
   - Potential loss of information during chunking

2. **Dependency on External AI Service**
   - Performance tied to Gemini API reliability
   - Potential latency in response generation
   - Cost implications for extensive use

3. **Inherent AI Limitations**
   - Risk of hallucination or misinterpreting context
   - Potential bias in response generation
   - Lack of absolute legal interpretation guarantee



## 🗺️ Development Roadmap

### Short-term Improvements
- Enhance semantic search algorithm
- Implement more granular confidence scoring
- Add support for multiple document types

### Long-term Vision
- Multi-model ensemble approach
- Advanced named entity recognition
- Cross-document reference tracking

## 🐛 Troubleshooting

### Common Issues
- API Key Configuration
- Dependency Version Conflicts
- PDF Parsing Errors

### Recommended Debugging
- Check Environment variable configuration
- Verify Python and package versions
- Use verbose logging mode

## 🤝 Contributing
1. Fork the repository
2. Create feature branch
3. Commit changes
4. Push to branch
5. Create pull request



## 📞 Contact
moseskiran006@gmail.com

---

**Disclaimer**: This tool is an AI-assisted system and should not replace professional legal consultation.
