import os
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'

from flask import Flask, request, render_template, redirect, url_for, flash
import fitz  # PyMuPDF
import nltk
nltk.download('punkt')

from transformers import BertTokenizer, BertModel
import torch
import logging

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.secret_key = 'your_secret_key'  
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  


os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)


nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

# Logging setup
logging.basicConfig(level=logging.INFO)

def extract_text_from_pdf(pdf_path):
    try:
        doc = fitz.open(pdf_path)
        text = ""
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            text += page.get_text()
        doc.close()
        return text
    except Exception as e:
        logging.error(f"Error extracting text from PDF: {e}")
        return ""

def preprocess_text(text):
    tokens = word_tokenize(text.lower())
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word.isalnum() and word not in stop_words]
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return tokens

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/model')
def model_page():
    return render_template('index.html', score="", feedback=None, message=None)

@app.route('/evaluate', methods=['POST'])
def evaluate():
    if 'keywordsFile' not in request.files or 'studentFile' not in request.files:
        flash('No file part')
        return redirect(request.url)
    
    keywords_file = request.files['keywordsFile']
    student_file = request.files['studentFile']
    
    if keywords_file.filename == '' or student_file.filename == '':
        flash('No selected file')
        return redirect(request.url)
    
    if keywords_file and keywords_file.filename.endswith('.pdf') and student_file and student_file.filename.endswith('.pdf'):
        keywords_path = os.path.join(app.config['UPLOAD_FOLDER'], 'keywords.pdf')
        student_path = os.path.join(app.config['UPLOAD_FOLDER'], 'student_answer.pdf')
        
        keywords_file.save(keywords_path)
        student_file.save(student_path)
        
        
        keywords_text = extract_text_from_pdf(keywords_path)
        student_text = extract_text_from_pdf(student_path)
        
        if not keywords_text or not student_text:
            flash('Error reading the PDF files. Please try again.')
            return redirect(request.url)
        
        preprocessed_keywords = preprocess_text(keywords_text)
        preprocessed_student = preprocess_text(student_text)
        
        
        if len(preprocessed_student) < 50:
            return render_template('index.html', score="", feedback=None, message="The answer is too short. Please provide an answer with more than 50 words.")
        
       
        keywords_set = set(preprocessed_keywords)
        student_set = set(preprocessed_student)
        common_keywords = keywords_set.intersection(student_set)
        
        
        score = (len(common_keywords) / len(keywords_set)) * 10 
        
        
        feedback = {
            'found_keywords': list(common_keywords),
            'missing_keywords': list(keywords_set - student_set)
        }
        
        return render_template('index.html', score=f" Score: {score:.2f}/10.00", feedback=feedback, message=None)
    
    flash('Invalid file type. Please upload PDF files.')
    return redirect(request.url)

if __name__ == '__main__':
    app.run(debug=True)
