import os
import re
import json
import pickle
import logging
import hashlib
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass, asdict
from datetime import datetime
import numpy as np
import pandas as pd

# Core ML libraries
import torch
from transformers import (
    AutoTokenizer, AutoModel, AutoModelForCausalLM, 
    pipeline, BitsAndBytesConfig
)
from sentence_transformers import SentenceTransformer

# Vector stores and search
import faiss
import chromadb
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Text processing
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class DocumentChunk:
    """Represents a chunk of text with metadata"""
    chunk_id: str
    text: str
    chunk_size: int
    doc_source: str
    chunk_index: int
    metadata: Dict[str, Any]
    embedding: Optional[np.ndarray] = None

@dataclass
class RetrievalResult:
    """Represents a retrieval result with scoring"""
    chunk: DocumentChunk
    dense_score: float
    sparse_score: float
    combined_score: float
    rank: int

@dataclass
class RAGResponse:
    """Represents the final RAG response"""
    query: str
    answer: str
    confidence: float
    response_time: float
    retrieved_chunks: List[RetrievalResult]
    method: str = "RAG"
    sources: List[str] = None

class InputGuardrail:
    """Input validation and filtering"""
    
    def __init__(self):
        self.financial_keywords = {
            # Core financial terms
            'revenue', 'profit', 'income', 'earnings', 'assets', 'liabilities', 
            'cash', 'flow', 'margin', 'dividend', 'bookings', 'employees',
            'tax', 'investment', 'growth', 'fiscal', 'quarter', 'annual',
            'financial', 'accenture', 'company', 'business',
            # Stock and equity terms
            'shares', 'outstanding', 'stock', 'equity', 'shareholders', 'shareholder',
            'share', 'eps', 'diluted', 'basic', 'per', 'common', 'preferred',
            # Financial metrics and ratios
            'operating', 'net', 'gross', 'ebitda', 'roce', 'roa', 'roe',
            'debt', 'equity', 'ratio', 'turnover', 'return', 'yield',
            # Time periods and dates
            'q1', 'q2', 'q3', 'q4', 'fy', 'fiscal', 'year', 'month', 'period',
            '2020', '2021', '2022', '2023', '2024', 'august', 'september', 'october',
            # Financial statements
            'balance', 'sheet', 'statement', 'consolidated', 'segment', 'gaap',
            # Currencies and amounts
            'million', 'billion', 'thousand', 'dollar', 'dollars', 'usd', '$',
            # Business metrics (including plurals)
            'clients', 'contracts', 'services', 'technology', 'consulting',
            'operations', 'acquisition', 'acquisitions', 'organic', 'bookings', 'billings',
            # Additional plurals for common financial terms
            'revenues', 'profits', 'earnings', 'dividends', 'investments', 'liabilities'
        }
        
        self.harmful_patterns = [
            r'\b(hack|attack|malware|virus)\b',
            r'\b(password|login|credential)\b',
            r'\b(personal|private|confidential)\b'
        ]
    
    def validate_query(self, query: str) -> Tuple[bool, str]:
        """Validate input query"""
        query_lower = query.lower()
        
        # Check for harmful content
        for pattern in self.harmful_patterns:
            if re.search(pattern, query_lower):
                return False, "Query contains potentially harmful content"
        
        # Check if query is financial-related (strict but comprehensive)
        words = set(re.findall(r'\b\w+\b', query_lower))
        financial_matches = words.intersection(self.financial_keywords)
        
        # Strict financial domain checking
        if financial_matches:
            # Has financial keywords - definitely financial
            pass
        else:
            # No financial keywords - check for financial numeric/currency indicators
            # Only accept if numbers appear with clear financial context
            currency_patterns = re.findall(r'[$€£¥]\d+[\d,\.]*|\b\d+[\d,\.]*%\b', query_lower)
            financial_numbers = re.findall(r'\b(?:million|billion|thousand|dollars?|usd)\b', query_lower)
            financial_numeric_context = re.findall(r'\b(?:cost|price|spend|spent|worth|value|paid)\s+[\$\d]|\b\d+[\d,\.]*\s+(?:million|billion|thousand|dollars?)\b', query_lower)
            
            if currency_patterns or financial_numbers or financial_numeric_context:
                # Has clear financial numeric indicators - likely financial
                pass
            else:
                # No financial keywords AND no clear financial numbers - reject
                return False, "Query appears to be outside financial domain"
        
        # Check query length
        if len(query.strip()) < 5:
            return False, "Query too short"
        
        if len(query.strip()) > 500:
            return False, "Query too long"
        
        return True, "Valid query"

class OutputGuardrail:
    """Output validation and filtering"""
    
    def __init__(self):
        self.min_confidence_threshold = 0.3
        self.hallucination_keywords = [
            'i think', 'i believe', 'probably', 'maybe', 'i assume',
            'my opinion', 'as far as i know', 'i guess'
        ]
    
    def validate_response(self, response: str, confidence: float, query: str) -> Tuple[bool, str, str]:
        """Validate and potentially modify output response"""
        
        # Check confidence threshold
        if confidence < self.min_confidence_threshold:
            modified_response = f"I have low confidence in this answer. {response}"
            return True, "Low confidence warning added", modified_response
        
        # Check for potential hallucination indicators
        response_lower = response.lower()
        for keyword in self.hallucination_keywords:
            if keyword in response_lower:
                modified_response = f"Note: This response contains uncertain language. {response}"
                return True, "Uncertainty warning added", modified_response
        
        # Check if response is too short
        if len(response.strip()) < 20:
            return False, "Response too short, possibly incomplete", response
        
        return True, "Response validated", response

class DocumentProcessor:
    """Handles document loading, cleaning, and chunking"""
    
    def __init__(self, chunk_sizes: List[int] = [100, 400]):
        self.chunk_sizes = chunk_sizes
        # Use a lightweight tokenizer for chunking
        try:
            # Try local Llama first
            self.tokenizer = AutoTokenizer.from_pretrained("models/Llama-3.1-8B-Instruct")
        except:
            try:
                # Fallback to HF Hub (requires gated access)
                self.tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
            except:
                # Final fallback to a simple tokenizer
                self.tokenizer = AutoTokenizer.from_pretrained("gpt2")
    
    def load_documents(self, doc_paths: List[str]) -> List[str]:
        """Load documents from multiple sources"""
        documents = []
        
        for doc_path in doc_paths:
            try:
                if doc_path.endswith('.txt'):
                    with open(doc_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                        documents.append(content)
                        logger.info(f"Loaded text document: {doc_path}")
                
                elif doc_path.endswith('.json'):
                    with open(doc_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        if isinstance(data, list):
                            # Handle Q&A format
                            content = self._process_qa_data(data)
                        else:
                            content = str(data)
                        documents.append(content)
                        logger.info(f"Loaded JSON document: {doc_path}")
                        
            except Exception as e:
                logger.error(f"Error loading document {doc_path}: {e}")
                continue
        
        return documents
    
    def _process_qa_data(self, qa_data: List[Dict]) -> str:
        """Convert Q&A data to text format"""
        text_content = []
        for item in qa_data:
            if 'question' in item and 'answer' in item:
                text_content.append(f"Q: {item['question']}")
                text_content.append(f"A: {item['answer']}")
                text_content.append("")
        return "\n".join(text_content)
    
    def clean_text(self, text: str) -> str:
        """Clean and preprocess text"""
        # Remove headers, footers, page numbers
        text = re.sub(r'Page \d+.*?\n', '', text, flags=re.IGNORECASE)
        text = re.sub(r'Header:.*?\n', '', text, flags=re.IGNORECASE)
        text = re.sub(r'Footer:.*?\n', '', text, flags=re.IGNORECASE)
        
        # Remove excessive whitespace
        text = re.sub(r'\n{3,}', '\n\n', text)
        text = re.sub(r' {2,}', ' ', text)
        
        # Remove special characters but keep financial symbols
        text = re.sub(r'[^\w\s\$\%\.\,\-\(\)]', ' ', text)
        
        return text.strip()
    
    def create_chunks(self, documents: List[str], doc_sources: List[str]) -> Dict[int, List[DocumentChunk]]:
        """Create chunks of different sizes from documents"""
        all_chunks = {size: [] for size in self.chunk_sizes}
        
        for doc_idx, (document, source) in enumerate(zip(documents, doc_sources)):
            cleaned_doc = self.clean_text(document)
            
            for chunk_size in self.chunk_sizes:
                chunks = self._chunk_by_tokens(cleaned_doc, chunk_size, source, doc_idx)
                all_chunks[chunk_size].extend(chunks)
        
        logger.info(f"Created chunks: {[(size, len(chunks)) for size, chunks in all_chunks.items()]}")
        return all_chunks
    
    def _chunk_by_tokens(self, text: str, max_tokens: int, source: str, doc_idx: int) -> List[DocumentChunk]:
        """Chunk text by token count with overlap"""
        sentences = sent_tokenize(text)
        chunks = []
        current_chunk = []
        current_tokens = 0
        overlap_ratio = 0.1  # 10% overlap
        
        chunk_idx = 0
        
        for sentence in sentences:
            # Count tokens in sentence
            sentence_tokens = len(self.tokenizer.encode(sentence, add_special_tokens=False))
            
            # If adding this sentence exceeds max_tokens, create a chunk
            if current_tokens + sentence_tokens > max_tokens and current_chunk:
                chunk_text = " ".join(current_chunk)
                chunk_id = self._generate_chunk_id(chunk_text, source, chunk_idx)
                
                chunk = DocumentChunk(
                    chunk_id=chunk_id,
                    text=chunk_text,
                    chunk_size=max_tokens,
                    doc_source=source,
                    chunk_index=chunk_idx,
                    metadata={
                        'doc_index': doc_idx,
                        'token_count': current_tokens,
                        'sentence_count': len(current_chunk)
                    }
                )
                chunks.append(chunk)
                
                # Create overlap for next chunk
                overlap_sentences = int(len(current_chunk) * overlap_ratio)
                current_chunk = current_chunk[-overlap_sentences:] if overlap_sentences > 0 else []
                current_tokens = sum(len(self.tokenizer.encode(s, add_special_tokens=False)) for s in current_chunk)
                chunk_idx += 1
            
            current_chunk.append(sentence)
            current_tokens += sentence_tokens
        
        # Handle remaining text
        if current_chunk:
            chunk_text = " ".join(current_chunk)
            chunk_id = self._generate_chunk_id(chunk_text, source, chunk_idx)
            
            chunk = DocumentChunk(
                chunk_id=chunk_id,
                text=chunk_text,
                chunk_size=max_tokens,
                doc_source=source,
                chunk_index=chunk_idx,
                metadata={
                    'doc_index': doc_idx,
                    'token_count': current_tokens,
                    'sentence_count': len(current_chunk)
                }
            )
            chunks.append(chunk)
        
        return chunks
    
    def _generate_chunk_id(self, text: str, source: str, chunk_idx: int) -> str:
        """Generate unique chunk ID"""
        content_hash = hashlib.md5(text.encode()).hexdigest()[:8]
        return f"{source}_{chunk_idx}_{content_hash}"

class EmbeddingManager:
    """Handles text embedding generation and management"""
    
    def __init__(self, model_path: str = "models/mxbai-embed-large-v1"):
        self.model_path = model_path
        self.model = None
        self.embedding_dim = None
        self._load_model()
    
    def _load_model(self):
        """Load the embedding model"""
        try:
            self.model = SentenceTransformer(self.model_path)
            self.embedding_dim = self.model.get_sentence_embedding_dimension()
            logger.info(f"Loaded embedding model: {self.model_path} (dim: {self.embedding_dim})")
        except Exception as e:
            logger.error(f"Error loading embedding model: {e}")
            # Fallback to smaller model
            self.model = SentenceTransformer('all-MiniLM-L6-v2')
            self.embedding_dim = self.model.get_sentence_embedding_dimension()
            logger.info(f"Loaded fallback embedding model (dim: {self.embedding_dim})")
    
    def embed_chunks(self, chunks: List[DocumentChunk]) -> List[DocumentChunk]:
        """Generate embeddings for chunks"""
        texts = [chunk.text for chunk in chunks]
        
        try:
            embeddings = self.model.encode(texts, batch_size=32, show_progress_bar=True)
            
            for chunk, embedding in zip(chunks, embeddings):
                chunk.embedding = embedding
            
            logger.info(f"Generated embeddings for {len(chunks)} chunks")
            
        except Exception as e:
            logger.error(f"Error generating embeddings: {e}")
            # Generate dummy embeddings as fallback
            for chunk in chunks:
                chunk.embedding = np.random.random(384)  # Standard dimension
        
        return chunks
    
    def embed_query(self, query: str) -> np.ndarray:
        """Generate embedding for a query"""
        try:
            return self.model.encode([query])[0]
        except Exception as e:
            logger.error(f"Error embedding query: {e}")
            return np.random.random(self.embedding_dim)

class VectorStoreManager:
    """Manages dense vector storage and retrieval"""
    
    def __init__(self, embedding_dim: int):
        self.embedding_dim = embedding_dim
        self.faiss_index = None
        self.chunk_map = {}  # Maps index -> chunk_id
        self.chromadb_client = None
        self.chromadb_collection = None
        self._setup_stores()
    
    def _setup_stores(self):
        """Setup vector stores"""
        # Setup FAISS
        self.faiss_index = faiss.IndexFlatIP(self.embedding_dim)  # Inner product similarity
        
        # Setup ChromaDB
        try:
            self.chromadb_client = chromadb.Client()
            self.chromadb_collection = self.chromadb_client.create_collection(
                name="financial_docs",
                metadata={"hnsw:space": "cosine"}
            )
            logger.info("Initialized vector stores (FAISS + ChromaDB)")
        except Exception as e:
            logger.warning(f"ChromaDB setup failed: {e}. Using FAISS only.")
    
    def add_chunks(self, chunks: List[DocumentChunk]):
        """Add chunks to vector stores"""
        embeddings = np.array([chunk.embedding for chunk in chunks])
        
        # Normalize embeddings for cosine similarity
        embeddings_normalized = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        
        # Add to FAISS
        self.faiss_index.add(embeddings_normalized)
        
        # Update chunk mapping
        start_idx = len(self.chunk_map)
        for i, chunk in enumerate(chunks):
            self.chunk_map[start_idx + i] = chunk.chunk_id
        
        # Add to ChromaDB if available
        if self.chromadb_collection:
            try:
                # Convert metadata to simple dict format for ChromaDB
                metadatas = []
                for chunk in chunks:
                    metadata = {
                        'doc_index': chunk.metadata.get('doc_index', 0),
                        'token_count': chunk.metadata.get('token_count', 0),
                        'sentence_count': chunk.metadata.get('sentence_count', 0),
                        'doc_source': chunk.doc_source,
                        'chunk_index': chunk.chunk_index
                    }
                    metadatas.append(metadata)
                
                self.chromadb_collection.add(
                    embeddings=embeddings_normalized.tolist(),
                    documents=[chunk.text for chunk in chunks],
                    metadatas=metadatas,
                    ids=[chunk.chunk_id for chunk in chunks]
                )
            except Exception as e:
                logger.warning(f"ChromaDB add failed: {e}")
        
        logger.info(f"Added {len(chunks)} chunks to vector stores")
    
    def search(self, query_embedding: np.ndarray, top_k: int = 10) -> List[Tuple[str, float]]:
        """Search for similar chunks"""
        query_normalized = query_embedding / np.linalg.norm(query_embedding)
        
        # Search with FAISS
        scores, indices = self.faiss_index.search(
            query_normalized.reshape(1, -1), top_k
        )
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx >= 0 and idx < len(self.chunk_map):  # Check for valid index
                chunk_id = self.chunk_map[idx]
                results.append((chunk_id, float(score)))
        
        return results

class SparseRetriever:
    """Handles sparse retrieval using BM25 and TF-IDF"""
    
    def __init__(self):
        self.bm25 = None
        self.tfidf_vectorizer = None
        self.tfidf_matrix = None
        self.chunk_map = {}  # Maps index -> chunk_id
        self.stemmer = PorterStemmer()
        self.stop_words = set(stopwords.words('english'))
    
    def build_index(self, chunks: List[DocumentChunk]):
        """Build sparse indexes"""
        processed_texts = []
        
        for i, chunk in enumerate(chunks):
            processed_text = self._preprocess_text(chunk.text)
            processed_texts.append(processed_text)
            self.chunk_map[i] = chunk.chunk_id
        
        # Build BM25 index
        tokenized_texts = [text.split() for text in processed_texts]
        self.bm25 = BM25Okapi(tokenized_texts)
        
        # Build TF-IDF index
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=10000,
            stop_words='english',
            ngram_range=(1, 2)
        )
        original_texts = [chunk.text for chunk in chunks]
        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(original_texts)
        
        logger.info(f"Built sparse indexes for {len(chunks)} chunks")
    
    def _preprocess_text(self, text: str) -> str:
        """Preprocess text for BM25"""
        # Tokenize and clean
        words = word_tokenize(text.lower())
        
        # Remove stopwords and stem
        processed_words = [
            self.stemmer.stem(word) 
            for word in words 
            if word.isalnum() and word not in self.stop_words
        ]
        
        return " ".join(processed_words)
    
    def search_bm25(self, query: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """Search using BM25"""
        processed_query = self._preprocess_text(query).split()
        scores = self.bm25.get_scores(processed_query)
        
        # Get top-k results
        top_indices = np.argsort(scores)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            if idx < len(self.chunk_map):
                chunk_id = self.chunk_map[idx]
                score = scores[idx]
                results.append((chunk_id, float(score)))
        
        return results
    
    def search_tfidf(self, query: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """Search using TF-IDF"""
        query_vector = self.tfidf_vectorizer.transform([query])
        similarities = cosine_similarity(query_vector, self.tfidf_matrix).flatten()
        
        # Get top-k results
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            if idx < len(self.chunk_map):
                chunk_id = self.chunk_map[idx]
                score = similarities[idx]
                results.append((chunk_id, float(score)))
        
        return results

class HybridRetriever:
    """Advanced RAG technique: Hybrid Search combining dense and sparse retrieval"""
    
    def __init__(self, vector_store: VectorStoreManager, sparse_retriever: SparseRetriever):
        self.vector_store = vector_store
        self.sparse_retriever = sparse_retriever
        self.chunk_store = {}  # chunk_id -> DocumentChunk
    
    def add_chunks(self, chunks: List[DocumentChunk]):
        """Add chunks to both retrievers"""
        # Store chunks for easy access
        for chunk in chunks:
            self.chunk_store[chunk.chunk_id] = chunk
        
        # Add to both retrievers
        self.vector_store.add_chunks(chunks)
        self.sparse_retriever.build_index(chunks)
    
    def retrieve(self, query: str, query_embedding: np.ndarray, 
                top_k: int = 10, alpha: float = 0.7) -> List[RetrievalResult]:
        """
        Hybrid retrieval with weighted score fusion
        alpha: weight for dense retrieval (1-alpha for sparse)
        """
        
        # Dense retrieval
        dense_results = self.vector_store.search(query_embedding, top_k * 2)
        
        # Sparse retrieval (BM25)
        sparse_results = self.sparse_retriever.search_bm25(query, top_k * 2)
        
        # Combine results
        all_chunk_ids = set()
        score_map = {}
        
        # Process dense results
        max_dense_score = max([score for _, score in dense_results], default=1.0)
        for chunk_id, score in dense_results:
            normalized_score = score / max_dense_score if max_dense_score > 0 else 0
            score_map[chunk_id] = {'dense': normalized_score, 'sparse': 0.0}
            all_chunk_ids.add(chunk_id)
        
        # Process sparse results
        max_sparse_score = max([score for _, score in sparse_results], default=1.0)
        for chunk_id, score in sparse_results:
            normalized_score = score / max_sparse_score if max_sparse_score > 0 else 0
            if chunk_id in score_map:
                score_map[chunk_id]['sparse'] = normalized_score
            else:
                score_map[chunk_id] = {'dense': 0.0, 'sparse': normalized_score}
                all_chunk_ids.add(chunk_id)
        
        # Calculate combined scores
        retrieval_results = []
        for chunk_id in all_chunk_ids:
            if chunk_id in self.chunk_store:
                scores = score_map[chunk_id]
                combined_score = alpha * scores['dense'] + (1 - alpha) * scores['sparse']
                
                result = RetrievalResult(
                    chunk=self.chunk_store[chunk_id],
                    dense_score=scores['dense'],
                    sparse_score=scores['sparse'],
                    combined_score=combined_score,
                    rank=0  # Will be set after sorting
                )
                retrieval_results.append(result)
        
        # Sort by combined score and assign ranks
        retrieval_results.sort(key=lambda x: x.combined_score, reverse=True)
        for i, result in enumerate(retrieval_results[:top_k]):
            result.rank = i + 1
        
        return retrieval_results[:top_k]

class ResponseGenerator:
    """Generates responses using the language model"""
    
    def __init__(self, model_path: str = "models/Llama-3.1-8B-Instruct", 
                 max_tokens: int = 4096, temperature: float = 0.3):
        self.model_path = model_path
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.tokenizer = None
        self.model = None
        self.generator = None
        self._load_model()
    
    def _load_model(self):
        """Load the language model"""
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            # Modern quantization config
            quantization_config = None
            try:
                # Respect sidebar setting if available
                import streamlit as st  # type: ignore
                if st.session_state.get('load_8bit', False):
                    quantization_config = BitsAndBytesConfig(
                        load_in_8bit=True,
                        llm_int8_threshold=6.0,
                        llm_int8_has_fp16_weight=False,
                        llm_int8_enable_fp32_cpu_offload=False
                    )
                # Default: no quantization (can be heavy if both models run)
            except Exception:
                pass
            
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                torch_dtype=torch.float16,
                quantization_config=quantization_config
            )
            # Skip manual GPU placement - let accelerate handle device placement
            
            # Create generation pipeline with proper response extraction
            self.generator = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                max_new_tokens=4096,
                do_sample=True,
                temperature=0.3,
                top_p=0.9,
                pad_token_id=self.tokenizer.eos_token_id,
                return_full_text=False  # Only return generated text, not prompt
            )
            
            logger.info(f"Loaded language model: {self.model_path}")
            
        except Exception as e:
            logger.error(f"Error loading language model: {e}")
            # Simple fallback for testing
            self.generator = None
    
    def generate_response(self, query: str, retrieved_chunks: List[RetrievalResult], 
                         max_tokens: int = None, temperature: float = None) -> Tuple[str, float]:
        """Generate response from query and retrieved context"""
        
        # Use config defaults if not provided
        if max_tokens is None:
            max_tokens = getattr(self, 'max_tokens', 4096)
        if temperature is None:
            temperature = getattr(self, 'temperature', 0.3)
        
        # Prepare context from retrieved chunks with token counting
        context_parts = []
        total_context_tokens = 0
        max_context_tokens = 12000  # Leave room for query and response
        
        for result in retrieved_chunks:
            chunk_text = f"Context: {result.chunk.text}"
            # Rough token estimation (4 chars per token)
            chunk_tokens = len(chunk_text) // 4
            
            if total_context_tokens + chunk_tokens > max_context_tokens:
                break
                
            context_parts.append(chunk_text)
            total_context_tokens += chunk_tokens
        
        context = "\n\n".join(context_parts)
        
        # Create prompt
        prompt = self._create_prompt(query, context)
        
        # Check total prompt length
        prompt_tokens = len(prompt) // 4  # Rough estimation
        if prompt_tokens > 15000:  # Leave room for response
            # Truncate context if needed
            max_context_length = (15000 - len(query) // 4 - 200) * 4  # Convert back to chars
            if len(context) > max_context_length:
                context = context[:max_context_length] + "...[truncated]"
                prompt = self._create_prompt(query, context)
        
        # Generate response
        if self.generator:
            try:
                # Generate response with pipeline
                outputs = self.generator(
                    prompt, 
                    max_new_tokens=max_tokens,
                    temperature=temperature,
                    do_sample=True,
                    top_p=0.9
                )
                
                # Extract generated text (should already exclude prompt due to return_full_text=False)
                if isinstance(outputs, list) and len(outputs) > 0:
                    if isinstance(outputs[0], dict) and 'generated_text' in outputs[0]:
                        response = outputs[0]['generated_text'].strip()
                    else:
                        response = str(outputs[0]).strip()
                else:
                    response = str(outputs).strip() if outputs else ""
                
                # Clean up any remaining template artifacts if full text was still returned
                if response.startswith('<|begin_of_text|>'):
                    # If full text still returned, extract only the assistant response
                    assistant_start = response.find('<|start_header_id|>assistant<|end_header_id|>\n')
                    if assistant_start != -1:
                        response = response[assistant_start + len('<|start_header_id|>assistant<|end_header_id|>\n'):].strip()
                        # Remove any end tokens
                        if '<|eot_id|>' in response:
                            response = response.split('<|eot_id|>')[0].strip()
                        if '<|end_of_text|>' in response:
                            response = response.split('<|end_of_text|>')[0].strip()
                
                # Ensure we have a valid response
                if not response or len(response.strip()) < 5:
                    logger.warning("Generated response is empty or too short, using fallback")
                    return self._fallback_response(query, retrieved_chunks)
                
                # Calculate confidence based on retrieval scores
                avg_score = np.mean([r.combined_score for r in retrieved_chunks[:3]])
                confidence = min(avg_score * 1.2, 1.0)  # Scale and cap at 1.0
                
                return response, confidence
                
            except Exception as e:
                logger.error(f"Error generating response: {e}")
                return self._fallback_response(query, retrieved_chunks)
        else:
            return self._fallback_response(query, retrieved_chunks)
    
    def _create_prompt(self, query: str, context: str) -> str:
        """Create a structured prompt for Llama 3.1 Instruct"""
        system = "You are a helpful AI assistant specialized in financial analysis. Use the provided context to answer questions accurately and provide complete, detailed responses. Include relevant context and explanations in your answers. For numerical answers, provide the complete sentence or paragraph that contains the information, not just the number alone."
        user = f"Context:\n{context}\n\nQuestion: {query}"
        # Basic Llama 3 chat-style template
        prompt = f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n{system}<|eot_id|>" \
                 f"<|start_header_id|>user<|end_header_id|>\n{user}<|eot_id|>" \
                 f"<|start_header_id|>assistant<|end_header_id|>\n"
        return prompt
    
    def _fallback_response(self, query: str, retrieved_chunks: List[RetrievalResult]) -> Tuple[str, float]:
        """Fallback response when model is not available"""
        if retrieved_chunks:
            # Use the best chunk as basis for response
            best_chunk = retrieved_chunks[0].chunk.text
            response = f"Based on the available financial data: {best_chunk[:200]}..."
            confidence = retrieved_chunks[0].combined_score
        else:
            response = "I couldn't find relevant information to answer your question."
            confidence = 0.1
        
        return response, confidence

class RAGPipeline:
    """Main RAG pipeline orchestrator"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or self._default_config()
        
        # Initialize components
        self.input_guardrail = InputGuardrail()
        self.output_guardrail = OutputGuardrail()
        self.doc_processor = DocumentProcessor(
            chunk_sizes=self.config['chunk_sizes']
        )
        self.embedding_manager = EmbeddingManager(
            model_path=self.config['embedding_model_path']
        )
        self.vector_store = VectorStoreManager(
            embedding_dim=self.embedding_manager.embedding_dim
        )
        self.sparse_retriever = SparseRetriever()
        self.hybrid_retriever = HybridRetriever(
            vector_store=self.vector_store,
            sparse_retriever=self.sparse_retriever
        )
        self.response_generator = ResponseGenerator(
            model_path=self.config['generative_model_path'],
            max_tokens=self.config['max_response_tokens'],
            temperature=self.config['temperature']
        )
        
        # State tracking
        self.is_initialized = False
        self.chunk_store = {}
    
    def _default_config(self) -> Dict[str, Any]:
        """Default configuration"""
        return {
            'chunk_sizes': [100, 400],
            'embedding_model_path': 'models/mxbai-embed-large-v1',
            'generative_model_path': 'models/Llama-3.1-8B-Instruct',
            'top_k_retrieval': 10,
            'alpha_hybrid': 0.7,  # Weight for dense vs sparse retrieval
            'max_response_tokens': 4096,
            'temperature': 0.3,
            'context_length': 16000,
            'enable_guardrails': True
        }
    
    def initialize(self, document_paths: List[str]) -> bool:
        """Initialize the RAG pipeline with documents"""
        try:
            logger.info("Initializing RAG pipeline...")
            
            # Load and process documents
            documents = self.doc_processor.load_documents(document_paths)
            doc_sources = [os.path.basename(path) for path in document_paths]
            
            # Create chunks for both sizes
            all_chunks_by_size = self.doc_processor.create_chunks(documents, doc_sources)
            
            # Use the larger chunk size for main retrieval
            main_chunks = all_chunks_by_size[max(self.config['chunk_sizes'])]
            
            # Generate embeddings
            main_chunks = self.embedding_manager.embed_chunks(main_chunks)
            
            # Build indexes
            self.hybrid_retriever.add_chunks(main_chunks)
            
            # Store chunks for easy access
            for chunk in main_chunks:
                self.chunk_store[chunk.chunk_id] = chunk
            
            self.is_initialized = True
            logger.info(f"RAG pipeline initialized with {len(main_chunks)} chunks")
            return True
            
        except Exception as e:
            logger.error(f"Error initializing RAG pipeline: {e}")
            return False
    
    def query(self, user_query: str) -> RAGResponse:
        """Process a user query through the full RAG pipeline"""
        start_time = datetime.now()
        
        try:
            # Input validation
            if self.config['enable_guardrails']:
                is_valid, validation_msg = self.input_guardrail.validate_query(user_query)
                if not is_valid:
                    return RAGResponse(
                        query=user_query,
                        answer=f"Query validation failed: {validation_msg}",
                        confidence=0.0,
                        response_time=0.0,
                        retrieved_chunks=[],
                        sources=[]
                    )
            
            # Check if pipeline is initialized
            if not self.is_initialized:
                return RAGResponse(
                    query=user_query,
                    answer="RAG pipeline not initialized. Please load documents first.",
                    confidence=0.0,
                    response_time=0.0,
                    retrieved_chunks=[],
                    sources=[]
                )
            
            # Generate query embedding
            query_embedding = self.embedding_manager.embed_query(user_query)
            
            # Retrieve relevant chunks
            retrieved_results = self.hybrid_retriever.retrieve(
                query=user_query,
                query_embedding=query_embedding,
                top_k=self.config['top_k_retrieval'],
                alpha=self.config['alpha_hybrid']
            )
            
            # Generate response
            response_text, confidence = self.response_generator.generate_response(
                query=user_query,
                retrieved_chunks=retrieved_results,
                max_tokens=self.config['max_response_tokens']
            )
            
            # Output validation
            if self.config['enable_guardrails']:
                is_valid, validation_msg, modified_response = self.output_guardrail.validate_response(
                    response_text, confidence, user_query
                )
                if is_valid and modified_response != response_text:
                    response_text = modified_response
            
            # Calculate response time
            response_time = (datetime.now() - start_time).total_seconds()
            
            # Extract sources
            sources = list(set([r.chunk.doc_source for r in retrieved_results]))
            
            return RAGResponse(
                query=user_query,
                answer=response_text,
                confidence=confidence,
                response_time=response_time,
                retrieved_chunks=retrieved_results,
                sources=sources
            )
            
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            response_time = (datetime.now() - start_time).total_seconds()
            
            return RAGResponse(
                query=user_query,
                answer=f"Error processing query: {str(e)}",
                confidence=0.0,
                response_time=response_time,
                retrieved_chunks=[],
                sources=[]
            )
    
    def save_pipeline(self, save_path: str):
        """Save the pipeline state"""
        try:
            state = {
                'config': self.config,
                'is_initialized': self.is_initialized,
                'chunk_count': len(self.chunk_store)
            }
            
            with open(save_path, 'wb') as f:
                pickle.dump(state, f)
            
            logger.info(f"Pipeline state saved to {save_path}")
            
        except Exception as e:
            logger.error(f"Error saving pipeline: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get pipeline statistics"""
        return {
            'is_initialized': self.is_initialized,
            'chunk_count': len(self.chunk_store),
            'embedding_dimension': self.embedding_manager.embedding_dim,
            'config': self.config
        }

# Utility functions for easy usage

def create_rag_pipeline(document_paths: List[str], config: Dict[str, Any] = None) -> RAGPipeline:
    """Create and initialize a RAG pipeline"""
    pipeline = RAGPipeline(config)
    
    if pipeline.initialize(document_paths):
        logger.info("RAG pipeline created successfully")
        return pipeline
    else:
        logger.error("Failed to create RAG pipeline")
        return None

def quick_query(pipeline: RAGPipeline, query: str) -> str:
    """Quick query function for simple usage"""
    response = pipeline.query(query)
    return response.answer

# Example usage and testing
if __name__ == "__main__":
    # Example usage
    document_paths = [
        "data/docs_for_rag/financial_qa_rag.txt",
        "data/processed/consolidated_documents_20250806_102519.txt"
    ]
    
    # Create pipeline
    rag_pipeline = create_rag_pipeline(document_paths)
    
    if rag_pipeline:
        # Test queries
        test_queries = [
            "What was Accenture's total revenue for fiscal year 2024?",
            "How much did Accenture invest in Generative AI?",
            "What is the capital of France?"  # Irrelevant query test
        ]
        
        for query in test_queries:
            print(f"\nQuery: {query}")
            response = rag_pipeline.query(query)
            print(f"Answer: {response.answer}")
            print(f"Confidence: {response.confidence:.2f}")
            print(f"Response Time: {response.response_time:.2f}s")
            print(f"Sources: {response.sources}")
            print("-" * 50) 