<body>
  <h1>IMDB Movie Reviews Sentiment Analysis</h1>

  <h2>Overview</h2>
  <p>This project implements a sentiment analysis model using the IMDB dataset of 50,000 movie reviews. The model classifies reviews as either positive or negative based on the text content.</p>

  <h2>Dataset</h2>
  <p>The dataset used in this project is the <a href="https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews">IMDB Dataset of 50K Movie Reviews</a> available on Kaggle. It contains 50,000 movie reviews labeled as positive or negative.</p>

  <h3>License</h3>
  <p>The dataset is licensed under "other".</p>

  <h2>Requirements</h2>
  <p>To run this project, you need the following libraries:</p>
  <ul>
      <li>Pandas</li>
      <li>NumPy</li>
      <li>TensorFlow</li>
      <li>Scikit-learn</li>
  </ul>
  <p>You can install the required libraries using pip:</p>
  <pre><code>!pip install pandas numpy tensorflow scikit-learn</code></pre>

  <h2>Installation</h2>
  <ol>
      <li><strong>Set Kaggle Configuration:</strong> Set the Kaggle configuration directory.
          <pre><code>import os
os.environ["kaggle_config_dir"]="/path/to/your/kaggle/directory"</code></pre>
      </li>
      <li><strong>Download the Dataset:</strong> Use the Kaggle API to download the dataset.
          <pre><code>!kaggle datasets download -d lakshmi25npathi/imdb-dataset-of-50k-movie-reviews</code></pre>
      </li>
      <li><strong>Unzip the Dataset:</strong> Extract the contents of the downloaded zip file.
          <pre><code>!unzip /content/imdb-dataset-of-50k-movie-reviews.zip</code></pre>
      </li>
  </ol>

  <h2>Data Preparation</h2>
  <p>Load the dataset and preprocess it:</p>
  <pre><code>import pandas as pd
import numpy as np

df = pd.read_csv('/content/IMDB Dataset.csv')
df['sentiment'].replace('positive', 1, inplace=True)
df['sentiment'].replace('negative', 0, inplace=True)</code></pre>

  <h2>Data Splitting</h2>
  <p>Split the data into training and testing sets:</p>
  <pre><code>from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(df['review'], df['sentiment'], test_size=0.2, random_state=42)</code></pre>

  <h2>Text Tokenization</h2>
  <p>Tokenize the text data:</p>
  <pre><code>from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

tokenizer = Tokenizer(num_words=5000)
tokenizer.fit_on_texts(df['review'])

X_train = tokenizer.texts_to_sequences(X_train)
X_test = tokenizer.texts_to_sequences(X_test)

X_train = pad_sequences(X_train, maxlen=100)
X_test = pad_sequences(X_test, maxlen=100)</code></pre>

  <h2>Model Building</h2>
  <p>Build the LSTM model:</p>
  <pre><code>from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense

model = Sequential()
model.add(Embedding(input_dim=5000, output_dim=128, input_length=100))
model.add(LSTM( 128, dropout=0.2))
model.add(Dense(1, activation="sigmoid"))</code></pre>

  <h2>Model Compilation</h2>
  <p>Compile the model:</p>
  <pre><code>model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])</code></pre>

  <h2>Model Training</h2>
  <p>Train the model:</p>
  <pre><code>model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=64)</code></pre>

  <h2>Model Evaluation</h2>
  <p>Evaluate the model's performance:</p>
  <pre><code>loss, accuracy = model.evaluate(X_test, y_test)
print("loss", loss)
print("accuracy", accuracy)</code></pre>

  <h2>Predictive System</h2>
  <p>Build a function to predict sentiment for new reviews:</p>
  <pre><code>def predictive_system(new_review):
  new_review = tokenizer.texts_to_sequences([new_review])
  new_review = pad_sequences(new_review, maxlen=100)
  prediction = model.predict(new_review)
  if prediction[0][0] > 0.5:
      sentiment = "positive"
  else:
      sentiment = "negative"
  return sentiment</code></pre>

  <h2>Example Prediction</h2>
  <p>Test the predictive system:</p>
  <pre><code>new_review = "this movie is amazing"
sentiment = predictive_system(new_review)
print(sentiment)</code></pre>

  <h2>Conclusion</h2>
  <p>This project demonstrates how to perform sentiment analysis on movie reviews using an LSTM model. The model can be further improved by tuning hyperparameters and experimenting with different architectures.</p>

  <h2>Acknowledgments</h2>
  <ul>
      <li><a href="https://www.kaggle.com/">Kaggle</a> for providing the dataset.</li>
      <li><a href="https://www.tensorflow.org/">TensorFlow</a> for the deep learning framework.</li>
  </ul>

  <h2>License</h2>
  <p>This project is licensed under the MIT License - see the <a href="LICENSE">LICENSE</a> file for details.</p>
</body>
