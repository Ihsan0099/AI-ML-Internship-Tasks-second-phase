# 📰 News Topic Classifier using BERT  

## 📌 Overview  
This project implements a **News Topic Classifier** using **BERT (Bidirectional Encoder Representations from Transformers)**.  
The goal is to classify news headlines into categories such as *World, Sports, Business, and Sci/Tech* by fine-tuning a pre-trained transformer model.  

## 🎯 Objective  
- Fine-tune a transformer model (BERT) to classify news headlines into topic categories.  
- Achieve high accuracy and F1-score through training and evaluation.  
- Provide an interactive interface for users to test the model.  

## 📂 Dataset  
- **AG News Dataset** (available on [Hugging Face Datasets](https://huggingface.co/datasets/ag_news))  
- Contains **120,000 training samples** and **7,600 test samples**.  
- Four categories:  
  - World 🌍  
  - Sports ⚽  
  - Business 💼  
  - Sci/Tech 🔬  

## ⚙️ Steps & Instructions  
1. **Tokenization & Preprocessing**  
   - Use Hugging Face `transformers` for tokenizing news headlines.  
   - Handle padding and truncation for uniform input length.  

2. **Model Training**  
   - Fine-tune the **`bert-base-uncased`** model using the preprocessed dataset.  
   - Train with GPU acceleration (if available) for efficiency.  

3. **Evaluation**  
   - Evaluate the model using **Accuracy** and **F1-score**.  
   - Compare results with baseline models for benchmarking.  

4. **Deployment**  
   - Deploy the trained model using **Streamlit** or **Gradio**.  
   - Provide a simple web interface where users can enter news headlines and get predicted categories in real-time.  

## 🚀 Tech Stack  
- **Python**  
- **PyTorch**  
- **Hugging Face Transformers**  
- **Datasets (Hugging Face)**  
- **Scikit-learn** (for metrics)  
- **Streamlit / Gradio** (for deployment)  

## 📊 Expected Output  
- Trained model that classifies news into 4 categories with high accuracy.  
- Interactive UI for real-time classification of news headlines.  

## ▶️ How to Run  
1. Clone this repository:  
   ```bash
   git clone https://github.com/your-username/news-topic-classifier.git
   cd news-topic-classifier
