# ml-model-comparison
### Repository Name:  
**ml-model-comparison**  

### Description:  
This project compares the performance of **Random Forest (RFC)**, **Support Vector Machine (SVM)**, and **Multi-Layer Perceptron (MLP)** classifiers on a given dataset using **GridSearchCV** for hyperparameter tuning. The best models for each algorithm are selected and evaluated based on classification reports and **F1-score**.  

### Features:  
- **Preprocessing**: Standardizes the dataset using `StandardScaler`.  
- **Hyperparameter Tuning**: Utilizes `GridSearchCV` with cross-validation (`cv=5`).  
- **Model Training & Evaluation**:  
  - **Random Forest Classifier (RFC)**  
  - **Support Vector Machine (SVM)**  
  - **Multi-Layer Perceptron (MLP)**  
- **Performance Metrics**: Generates **classification reports** and computes the **F1-score** for each model.  

### Dependencies:  
- `pandas`  
- `scikit-learn`  
- `numpy`  

### Usage:  
1. Install dependencies:  
   ```bash
   pip install pandas scikit-learn numpy
   ```  
2. Ensure you have a dataset (`example.csv`) with a `label` column for classification.  
3. Run the script:  
   ```bash
   python model_comparison.py
   ```  

### Output:  
- The script prints classification reports and **F1-scores** for all three models.  
- The best hyperparameters for each model are automatically selected via `GridSearchCV`.  

---

💡 **Note:**  
- The `param_grid` is currently set for **Random Forest** and does not apply to **SVM** and **MLP**. Consider customizing the hyperparameters for each model separately for better tuning.  
- Running `GridSearchCV` for all three models sequentially may be slow. You may want to parallelize or reduce the search space for efficiency.  

Let me know if you need modifications! 🚀
