# Word Embedding

## 1. Introduction

Traditional word embeddings are typically trained using methods such as n-grams or bag-of-words. In this project, we build a simple neural network to train a word embedding matrix, using mean squared error as the loss function and targeting numerical values such as 'price', 'longitude', and 'latitude' from an Airbnb dataset. Our objective is to understand how word embeddings work and to evaluate their effectiveness in the context of Airbnb listings.

## 2. Data Description

The data is downloaded from [Inside Airbnb](https://insideairbnb.com/get-the-data/). For this model, we used Airbnb listings from San Diego and Los Angeles.

Preprocessing: We collected all the words from the following columns: ['name', 'description', 'neighborhood_overview', 'host_about', 'room_type', 'amenities']. We then created a vocabulary from all the words in the dataset and assigned an index to each word. Each input is represented as a one-hot encoded vector, where for each row, if a word appears in one of the columns mentioned above, the corresponding index in the one-hot encoded vector is incremented by 1.

Our target is a vector with values from the columns ['price', 'longitude', 'latitude', 'accommodates', 'bathrooms', 'bedrooms', 'beds', 'minimum_nights', 'review_scores_rating', 'number_of_reviews']. This setup creates a context for training the word embeddings, as it associates words with numerical attributes from the listings. The idea is to learn meaningful embeddings that capture the relationships between words and these numerical features. For example, certain words in the description might be more frequently associated with higher prices or specific types of accommodation, allowing the model to understand these patterns.

## 3. Methods

Our model is a straightforward neural network with two fully connected layers:
```
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(VOCAB_SIZE, EMBEDDING_DIM)
        self.fc2 = nn.Linear(EMBEDDING_DIM, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

```

The ReLU activation function is applied after each layer, introducing non-linearity into the model. This non-linearity allows the model to learn more complex relationships within the data compared to a purely linear model.
For training, we use the Adam optimizer and the Mean Squared Error (MSE) loss function. The MSE loss is appropriate for our task as we are predicting continuous numerical values

## 4. Results

- **Train and Validation loss**: 

We trained the model for 10 epochs. During training, the training loss decreased significantly with each epoch, indicating that the model was successfully learning to associate the words with the numerical values in the training data. However, we observed that the validation loss did not decrease after a certain point, which led us to stop training at 10 epochs. This suggests that while the model was able to capture relationships in the training data, it struggled to generalize to the validation data.
![Train Val loss](word_embed\train_val_loss.png)


- **Words' Relationship**:

The word embedding matrix was able to train and establish clear relationships between words. For example, the top 10 closest words to 'san' are:
```
find_closest_words('san', word_to_idx, idx_to_word, model, 10)
Closest words to 'san':
Index: 30882, Word: diego
Index: 16138, Word: mission
Index: 19898, Word: sd
Index: 11757, Word: italy
Index: 14253, Word: jolla
Index: 29222, Word: gaslamp
Index: 21133, Word: sdsu
Index: 21127, Word: 805
Index: 2976, Word: chula
Index: 12360, Word: petco

find_closest_words('angeles', word_to_idx, idx_to_word, model, 10)
Closest words to 'angeles':
Index: 19263, Word: los
Index: 35594, Word: la
Index: 1913, Word: venice
Index: 16294, Word: universal
Index: 23689, Word: manhattan
Index: 34720, Word: lax
Index: 28736, Word: monica
Index: 15035, Word: santa
Index: 6174, Word: hollywood
Index: 22787, Word: l
```

The examples above show that the word embedding can capture semantic similarity. Words that frequently appear together in similar contexts tend to be close in the embedding space. For instance, 'san' and 'diego' are close because 'San Diego' is a commonly used name for the city in the dataset. Similarly, 'angeles' and 'los' are close because 'Los Angeles' is a well-known city name.

Interestingly, the closest words to 'students' are:
```
find_closest_words('students', word_to_idx, idx_to_word, model, 10)
Closest words to 'students':
Index: 26911, Word: professionals
Index: 33364, Word: adult
Index: 5022, Word: therapist
Index: 15323, Word: solarium
Index: 13975, Word: actors
Index: 23850, Word: roles
Index: 30057, Word: meadows
Index: 9512, Word: cpk
Index: 1533, Word: seem
Index: 12999, Word: chances
```
- **Price predicting**:

Although the model fails to map words to numerical values perfectly, the scatter plot of actual price versus predicted price shows a clear positive correlation, with most points close to the perfect fit line. The model tends to underpredict the price when the actual price is high.
![Predicted vs Actual Price](word_embed\predict_vs_actual.png)

## 5. Discusion and Improvement

Our vocabulary is not ideal as it still contains unexpected words such as emojis and variations of words (e.g., "student" vs. "students"), which significantly increase the size of the vocabulary and dilute the relationships between words.

We can explore incorporating positional encoding to maintain the meaning and patterns of words more effectively. Additionally, our dataset can be further cleaned to be more consistent, with fewer random entries and duplicates.
