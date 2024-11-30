import sys
import os
import re
from typing import List, Dict, Any
import numpy as np
import logging

import google.generativeai as genai
from dotenv import load_dotenv
import PyPDF2
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class RobustLegalDocumentQA:
    def __init__(self, pdf_path: str, log_level: str = 'INFO'):
        
        logging.basicConfig(
            level=getattr(logging, log_level.upper()),
            format='%(asctime)s - %(levelname)s: %(message)s'
        )
        self.logger = logging.getLogger(__name__)

        
        load_dotenv()

        
        try:
            genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))
            self.model = genai.GenerativeModel('gemini-pro')
        except Exception as e:
            self.logger.error(f"Failed to initialize Gemini API: {e}")
            raise

        
        self.document_text = self.extract_pdf_text(pdf_path)
        self.text_chunks = self.advanced_text_chunking(self.document_text)
        
        
        self.vectorizer = TfidfVectorizer(stop_words='english')
        self.chunk_vectors = self.vectorizer.fit_transform(self.text_chunks)

    def extract_pdf_text(self, pdf_path: str) -> str:
        """
        Enhanced PDF text extraction with error handling and cleaning.
        """
        try:
            with open(pdf_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                full_text = ''
                for page in reader.pages:
                    page_text = page.extract_text() or ''
                
                    page_text = self.clean_text(page_text)
                    full_text += page_text + '\n'
            
            self.logger.info(f"Successfully extracted text from {pdf_path}")
            return full_text
        except Exception as e:
            self.logger.error(f"Error extracting PDF text: {e}")
            raise

    @staticmethod
    def clean_text(text: str) -> str:
        """
        Advanced text cleaning method.
        """
        
        text = re.sub(r'\s+', ' ', text).strip()
        
        
        text = re.sub(r'[^\w\s.,;:()-]', '', text)
        
        return text

    def advanced_text_chunking(self, text: str, chunk_size: int = 1500, overlap: int = 300) -> List[str]:
        
        

        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        chunks = []
        current_chunk = []
        current_length = 0

        for sentence in sentences:
            
            current_chunk.append(sentence)
            current_length += len(sentence)

            
            if current_length >= chunk_size:
                chunks.append(' '.join(current_chunk))
                
                
                current_chunk = current_chunk[-int(overlap/50):]
                current_length = len(' '.join(current_chunk))

        
        if current_chunk:
            chunks.append(' '.join(current_chunk))

        return chunks

    def semantic_search(self, question: str, top_k: int = 3) -> List[str]:
        """
        Advanced semantic search using TF-IDF and cosine similarity.
        """
    
        question_vector = self.vectorizer.transform([question])
        
        
        similarities = cosine_similarity(question_vector, self.chunk_vectors)[0]
        
        
        top_indices = similarities.argsort()[-top_k:][::-1]
        relevant_chunks = [self.text_chunks[i] for i in top_indices]
        
        return relevant_chunks

    def generate_comprehensive_answer(self, question: str) -> Dict[str, Any]:
        """
        Generate a comprehensive answer with multiple layers of processing.
        """
        try:
        
            relevant_chunks = self.semantic_search(question)
            
            
            prompt = f"""
            Context Chunks: {' || '.join(relevant_chunks)}

            Question: {question}

            Instructions:
            1. Analyze the context chunks thoroughly
            2. Provide a precise, factual answer
            3. If uncertain, clearly state the limitations
            4. Cite the most relevant context chunk
            5. Maintain professional, legal-focused language
            """

            
            response = self.model.generate_content(prompt, 
                generation_config=genai.types.GenerationConfig(
                    temperature=0.3,  
                    max_output_tokens=5000
                )
            )

            
            return {
                'answer': response.text,
                'confidence': self._calculate_confidence(response.text),
                'relevant_chunks': relevant_chunks
            }

        except Exception as e:
            self.logger.error(f"Answer generation failed: {e}")
            return {
                'answer': "Unable to generate a reliable answer.",
                'confidence': 0.0,
                'relevant_chunks': []
            }

    def _calculate_confidence(self, answer: str) -> float:
        """
        Calculate a basic confidence score for the generated answer.
        """
        
        definitive_indicators = ['clearly states', 'explicitly mentions', 'according to']
        confidence = 0.5  

        
        if len(answer) > 100 and any(ind in answer.lower() for ind in definitive_indicators):
            confidence += 0.3

        
        if 'cannot find' in answer.lower() or 'uncertain' in answer.lower():
            confidence -= 0.2

        return max(0.0, min(confidence, 1.0))  # Clip between 0 and 1

def main():
    
    if len(sys.argv) != 3:
        print("Usage: python3 question-answering.py /path/to/document.pdf 'Your question'")
        sys.exit(1)
    
    pdf_path, question = sys.argv[1], sys.argv[2]
    
    
    if not os.path.exists(pdf_path):
        print(f"Error: PDF file {pdf_path} does not exist.")
        sys.exit(1)
    
    # Perform QA
    qa_system = RobustLegalDocumentQA(pdf_path)
    result = qa_system.generate_comprehensive_answer(question)
    
    # Enhanced output
    print("📄 Answer:", result['answer'])
    #print("\n🔍 Confidence Score:", f"{result['confidence']*100:.2f}%")
    print("\n📍 Context Chunks:")
    for i, chunk in enumerate(result['relevant_chunks'], 1):
        print(f"Chunk {i}: {chunk[:1000]}...")

if __name__ == "__main__":
    main()