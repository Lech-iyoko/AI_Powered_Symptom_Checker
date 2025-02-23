# AI_Powered_Symptom_Checker

This project aims to develop a **symptom checker assitant** that assists with **predicting potential conditions or outcomes based on textual descriptions of symptoms and medical history**. By analyzing patient data, the model supports decision-making processes in healthcare, offering insights into likely diagnoses or next steps. This project demonstrates the application of AI in healthcare, providing a foundational framework for a scalable **Healthcare Assistant**.

### Dataset
* Name: MedText Dataset
* Size: 1.4k rows, 2 columns
* Columns:
  - Prompt: Text describing patient symptoms, history, or complaints.
  - Completion: The most likely outcome or diagnosis for the given prompt.
* Data Content: Includes descriptions of the top 100 most common diseases and 30 most common injuries.

### Objective
Build an AI-powered symptom checker that can:
* Predict potential conditions or outcomes based on input symptoms.
* Efficiently preprocess and handle textual data for better insights.
* Serve as a foundational model for further development in healthcare applications.

### Workflow
1. Data Collection and Preprocessing:
Handle missing data, duplicates, and inconsistencies.

2. Clean and normalize text by:
Removing punctuation, special characters, and stopwords.
Expanding contractions and removing accents.
Converting text to lowercase and normalizing whitespace.
Tokenize text data for vectorization.
Tools: Python libraries such as pandas, textacy, nltk, and spacy.

3. Exploratory Data Analysis (EDA)
Goals:
Identify patterns and distributions in the data.
Analyze text length, token frequencies, and common entities.
Visualize key metrics using charts and graphs.
Tools: matplotlib, seaborn, spacy.

4. Feature Engineering
Techniques:
Convert text to numerical formats using:
Term Frequency-Inverse Document Frequency (TF-IDF).
Word embeddings like Word2Vec or GloVe.
Explore n-grams for phrase-level features.

5. Model Development
Approach:
Train models like:
Logistic Regression
Random Forest
Neural Networks (e.g., RNN, LSTM, or Transformers like BERT)
Optimize for accuracy, precision, recall, and generalizability.
Tools: scikit-learn, TensorFlow, PyTorch.

6. Model Evaluation
Metrics:
Accuracy
Precision, Recall, F1-score
ROC-AUC for classification performance
Comparison: Evaluate and compare models to select the most reliable one.

### Dependencies
* Python 3.8+
* Libraries:
  - pandas
  - numpy
  - textacy
  - scikit-learn
  - matplotlib, seaborn
  - nltk, spacy
 
### Future Directions
* Expand Dataset: Incorporate additional medical data for better coverage of diseases and symptoms.
* Integration: Develop a user interface or API for real-world application.
* Explainability: Use tools like LIME or SHAP to make predictions more interpretable for healthcare providers.
