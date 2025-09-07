# 🏠 Multimodal Housing Price Prediction

## 📌 Task Overview
**Task 3: Multimodal ML – Housing Price Prediction Using Images + Tabular Data**

The objective of this project is to **predict housing prices** using both **structured tabular data** (e.g., number of bedrooms, bathrooms, square footage) and **unstructured image data** (house photos).  
This task demonstrates how to build a **multimodal machine learning model** that can learn from multiple data sources simultaneously.

---

## 📊 Dataset
- **Tabular Data**: Housing sales dataset (features: `bed`, `bath`, `sqft`, `price`)  
- **Image Data**: House images (64x64 RGB format)  
- **Target Variable**: `price` (regression)

---

## ⚙️ Implementation Steps
1. **Data Preparation**
   - Split dataset into training and testing sets (75/25 split)
   - Normalize tabular features
   - Resize and scale images
   - Extract target variable (`price`) for regression

2. **Model Architecture**
   - **ANN (MLP)** for tabular data (`bed`, `bath`, `sqft`)
   - **CNN** for image data (Conv2D + MaxPooling + Flatten layers)
   - **Concatenation Layer** to merge ANN and CNN features
   - **Dense Layers** on top for final regression output

3. **Training**
   - Optimizer: `Adam`
   - Loss Function: `Mean Squared Error (MSE)`
   - Epochs: 100
   - Batch Size: 100

4. **Evaluation**
   - Performance metrics:
     - **Mean Absolute Error (MAE)**
     - **Root Mean Squared Error (RMSE)**
   - Plotted training vs validation loss

---

## 📈 Results
- Model successfully trained on combined tabular + image data  
- Evaluation metrics calculated:
  - **MAE**: Lower values indicate better accuracy  
  - **RMSE**: Penalizes larger prediction errors  

*(Exact numbers depend on dataset and training run)*

---

## 📦 Dependencies
Make sure to install the following:
```bash
pip install tensorflow scikit-learn matplotlib numpy
