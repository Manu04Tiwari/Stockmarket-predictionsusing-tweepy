# Stock Movement Prediction Using Tweepy 
![image](https://github.com/Manu04Tiwari/Stockmarket-predictionsusing-tweepy/blob/main/image.png)
## Description
This project focuses on developing a machine learning model to predict stock price movements by analyzing data scraped from social media platforms, particularly Twitter. By leveraging Natural Language Processing (NLP) techniques, the model extracts sentiment, keywords, and trends from user-generated content, which are then used to forecast stock price direction or changes. The system aims to provide actionable insights for traders and analysts by capturing market sentiment in real-time.

---

## Features
- **Data Scraping**: Extract real-time tweets related to stocks using the Twitter API.
- **Sentiment Analysis**: Perform sentiment classification (positive, neutral, negative) on the scraped data.
- **Feature Engineering**: Extract features such as tweet volume, sentiment scores, and keyword frequency.
- **Stock Price Prediction**: Predict stock movements (up, down, or percentage change) using machine learning models.
- **Visualization**: Provide insights through sentiment heatmaps, keyword trends, and prediction confidence scores.
![image](https://github.com/Manu04Tiwari/Stockmarket-predictionsusing-tweepy/blob/main/eg.output.PNG)
---

## Project Workflow
1. **Data Collection**:
   - Scraped tweets using the Twitter API filtered by hashtags and ticker symbols (e.g., $AAPL, $TSLA).
   - Additional data sources (e.g., Reddit discussions, news articles) can be incorporated for enhanced accuracy.

2. **Data Preprocessing**:
   - Removed noise (e.g., URLs, mentions, special characters) from tweets.
   - Tokenized text and applied sentiment scoring using NLP models (e.g., Vader, BERT).

3. **Feature Extraction**:
   - Extracted features like sentiment polarity, tweet volume, and keyword density.
   - Integrated historical stock price data for time-series analysis.

4. **Model Training and Evaluation**:
   - Trained models such as Logistic Regression, Random Forest, LSTM, and Transformers.
   - Evaluated model performance using metrics like accuracy, precision, recall, F1-score, and MAE.

5. **Results and Insights**:
   - Highlighted the correlation between social sentiment and stock price movements.
   - Identified trends and keyword triggers influencing stock volatility.

6. **Future Enhancements**:
   - Integrate Reddit, news articles, and economic indicators.
   - Expand to real-time prediction pipelines and multilingual sentiment analysis.

---

## How to Run the Project

### Prerequisites
- **Programming Language**: Python (>= 3.8)
- **Libraries**: Install the following Python packages:
  ```bash
  pip install pandas numpy scikit-learn nltk tweepy transformers keras

 ### Steps 
 ## To clone
    git clone https://github.com/Manu04Tiwari/stock-prediction
    cd stock-prediction

## Technologies Used
- **Programming**: Python
- **NLP Tools**: NLTK, Vader, Transformers (e.g., BERT, FinBERT)
- **Machine Learning**: scikit-learn, TensorFlow/Keras
- **Data Collection**: Twitter API, Tweepy
- **Visualization**: Matplotlib, Seaborn, Streamlit

---
## Dataset
- **Sources**: Twitter data (scraped using hashtags and ticker symbols).
- **Preprocessing Steps**: Noise removal, tokenization, and feature extraction.
- **Size**: Approximately 50,000 tweets for training and testing.
![image](https://github.com/Manu04Tiwari/Stockmarket-predictionsusing-tweepy/blob/main/vedar%20model%20text%20data.jpg)


---

## Future Enhancements
1. Integrate more data sources (e.g., Reddit, news articles, financial reports).
2. Implement real-time prediction pipelines with live dashboards.
3. Introduce advanced NLP techniques (e.g., multilingual models, contextual sentiment analysis).
4. Expand predictions to include volatility and sector-wide trends.

---

## Model Details
- **Baseline Models**: Logistic Regression, Random Forest.
- **Advanced Models**: LSTM (time-series analysis), BERT (sentiment analysis).
- **Evaluation Metrics**:
  - **Classification**: Accuracy, Precision, Recall, F1-score.
  - **Regression**: MAE, RMSE, R².

---
## Contributors
- **Manu Tiwari**

---

## Contact
For questions or feedback, please contact:
- **Email**: [marttiwari8219@gmail.com]
- **GitHub**: [https://github.com/Manu04Tiwari](https://github.com/Manu04Tiwari)
