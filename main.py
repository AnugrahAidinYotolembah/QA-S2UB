#memasukkan library python
import streamlit as st
from transformers import BertTokenizerFast, pipeline
from PyPDF2 import PdfReader
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Inisialisasi tokenizer dan pipeline
tokenizer = BertTokenizerFast.from_pretrained('Wikidepia/indobert-lite-squad')
qa_pipeline = pipeline("question-answering", model="Wikidepia/indobert-lite-squad", tokenizer=tokenizer, device=-1)

# Fungsi untuk membaca teks dari file PDF
def read_pdf(file):
    pdf_reader = PdfReader(file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

# Fungsi untuk mendapatkan jawaban panjang
def get_long_answer(context, question, max_length):
    answers = []
    chunk_size = 512  # Ukuran maksimum token per chunk
    total_length = 0  # Variabel untuk menghitung total panjang jawaban
    
    for i in range(0, len(context), chunk_size):
        chunk = context[i:i + chunk_size]
        result = qa_pipeline({'context': chunk, 'question': question})
        answer = result['answer']
        
        # Tambahkan panjang jawaban ke total panjang
        answer_length = len(answer.split())
        if total_length + answer_length <= max_length:
            answers.append(answer)
            total_length += answer_length
        else:
            break
    
    long_answer = " ".join(answers)
    return long_answer

# Fungsi untuk mencari kalimat relevan berdasarkan cosine similarity
def find_relevant_sentence(context, question):
    sentences = context.split('.')
    vectorizer = TfidfVectorizer()
    
    # Menggabungkan kalimat untuk TF-IDF
    tfidf_matrix = vectorizer.fit_transform(sentences + [question])
    cosine_similarities = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1])
    
    # Mengambil indeks kalimat dengan cosine similarity tertinggi
    best_sentence_index = np.argmax(cosine_similarities)
    return sentences[best_sentence_index].strip()

# Judul aplikasi
st.title("QA program")
st.markdown("<h4>Create by Aidin's and friends</h4>", unsafe_allow_html=True)
# Input file PDF
uploaded_file = st.file_uploader("Unggah file PDF", type=["pdf"])

# Input pertanyaan
question = st.text_input("Masukkan pertanyaan:")

# Input panjang jawaban maksimum
max_length = st.number_input("Panjang maksimum jawaban (dalam kata)", min_value=1, value=50, step=1)

if uploaded_file is not None and question:
    # Membaca teks dari PDF
    context = read_pdf(uploaded_file)
    
    # Memastikan ada teks untuk digunakan sebagai konteks
    if context:
        # Mencari kalimat relevan berdasarkan cosine similarity
        relevant_sentence = find_relevant_sentence(context, question)
        
        # Mendapatkan jawaban panjang
        long_answer = get_long_answer(context, question, max_length)
        
        # Menampilkan hasil
        st.subheader("Hasil Tanya Jawab:")
        st.write("Jawaban :", long_answer)
        
        # Menampilkan kalimat relevan
        st.subheader("Kalimat Relevan:")
        st.write(relevant_sentence)
    else:
        st.error("Tidak dapat mengekstrak teks dari PDF.")
