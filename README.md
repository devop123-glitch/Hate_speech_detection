# Hate_speech_detection
Real-time Facial Emotion Detection using deep learning and machine learning:
Hate Speech Detection using NLP and Deep Learning
Introduction

This project focuses on building a hate speech detection system using natural language processing techniques and multiple modeling approaches. The goal is to classify text into hate speech or non-hate speech, using a combination of traditional machine learning and deep learning methods.

Instead of directly training models on raw text, the project emphasizes a step-by-step text preprocessing pipeline, followed by different feature extraction strategies and model comparisons. This makes it easier to understand how each stage contributes to the final performance.

Dataset

The dataset used in this project is a curated hate speech dataset obtained from Kaggle. It contains text data along with corresponding labels indicating whether the content is hate speech or not.

Two versions of the dataset are available:

A general dataset
A balanced dataset, which is used in this project to avoid bias toward any class

The main columns used are:

Content: the text data
Label: the target variable (hate / non-hate)
Overall Workflow

The project follows a structured pipeline:

Data loading
Text preprocessing
Tokenization and normalization
Feature extraction (BoW, TF-IDF, Word2Vec)
Model training (Naive Bayes, LSTM)
Evaluation using metrics and ROC curve
Data Loading

The dataset is downloaded using kagglehub and loaded into a Pandas DataFrame. After loading, basic inspection is performed using methods like:

head() to preview data
value_counts() to check class distribution
Text Preprocessing

A major part of this project is dedicated to cleaning and standardizing the text. The preprocessing is done in multiple clearly defined steps:

1. Lowercasing

All text is converted to lowercase to maintain consistency and avoid duplicate representations of the same word.

2. Removing HTML Tags

Some text entries may contain HTML elements. These are removed using BeautifulSoup to retain only meaningful textual content.

3. Removing URLs

Links and URLs are removed since they do not contribute to semantic meaning in most cases.

4. Chat Word Conversion

Informal chat abbreviations such as “LOL”, “BRB”, “OMG” are expanded into their full forms using a predefined dictionary. This helps in improving interpretability and model understanding.

5. Emoji Conversion

Emojis are converted into text descriptions (e.g., 😊 → “smiling_face”) using the emoji library, allowing them to be used as meaningful features.

6. Removing Punctuation

All punctuation marks are removed to simplify the text and reduce noise.

7. Spell Correction

Spelling mistakes are corrected using the pyspellchecker library. This step ensures that similar words are not treated as different tokens due to typos.

8. Tokenization

The cleaned text is split into individual words (tokens) using NLTK’s tokenizer.

9. Stopword Removal (Custom)

Standard stopwords are removed, but with an important modification:

Words like “not”, “no”, “never” are retained
Pronouns are also kept

This is important because such words carry strong contextual meaning in hate speech detection.

10. Lemmatization

Each word is reduced to its base form using WordNet Lemmatizer. This helps group similar words together (e.g., “running” → “run”).

11. Reconstructing Text

The processed tokens are joined back into sentences, creating a clean final text column used for modeling.

Feature Extraction

Multiple feature extraction techniques are used to represent the text numerically:

1. Bag of Words (BoW)
Uses word frequency counts
Includes unigrams and bigrams
Limited to top 5000 features
2. TF-IDF
Captures importance of words relative to documents
Reduces the weight of common words
3. Word2Vec
Trains a custom Word2Vec model using the dataset
Converts each word into a dense vector
Sentence vectors are created by averaging word vectors

This provides a more semantic representation compared to BoW and TF-IDF.

Model Training
1. Bernoulli Naive Bayes

Three separate models are trained using:

Word2Vec features
Bag of Words
TF-IDF

Each model is evaluated using:

Accuracy
Classification report

This helps compare how different feature representations affect performance.

2. LSTM (Deep Learning Model)

A deep learning model is built using TensorFlow/Keras:

Architecture:
Embedding layer (to convert words into dense vectors)
Three stacked LSTM layers
Dropout layer (to prevent overfitting)
Dense output layer with sigmoid activation
Training:
Binary cross-entropy loss
Adam optimizer
Trained for multiple epochs

This model captures sequential dependencies in text, making it more powerful for language understanding.

Evaluation

The models are evaluated using multiple metrics:

Accuracy
Precision, Recall, F1-score
Confusion Matrix

For the LSTM model, additional evaluation includes:

ROC Curve and AUC
ROC curve is plotted to visualize performance
AUC score is calculated to measure model quality
Threshold Optimization
The optimal classification threshold is selected using:
J = TPR - FPR

This helps improve classification decisions beyond the default 0.5 threshold.

Prediction Function

A helper function is created to predict whether a given sentence contains hate speech:

predict_hate("your input text here")

This function processes the input text and returns:

0 → Non-hate
1 → Hate speech
Key Learnings
Text preprocessing significantly impacts model performance
Keeping important stopwords (like “not”) improves results
Word2Vec captures semantic meaning better than simple frequency-based methods
LSTM performs better in understanding sequence and context
Threshold tuning can improve classification accuracy
Limitations
Word2Vec is trained only on the given dataset (limited vocabulary)
LSTM training is relatively basic (no hyperparameter tuning)
No cross-validation applied
Dataset size may limit generalization
How to Run
Install dependencies:
pip install pandas numpy nltk scikit-learn gensim tensorflow emoji beautifulsoup4 pyspellchecker textblob
Run the notebook step by step
Ensure NLTK resources are downloaded:

Conclusion

This project demonstrates a complete NLP pipeline—from raw text to model evaluation. By combining traditional machine learning techniques with deep learning, it provides a clear comparison of different approaches to hate speech detection.

The most important takeaway is that data preprocessing and feature engineering play a crucial role, often as important as the model itself.

Author
