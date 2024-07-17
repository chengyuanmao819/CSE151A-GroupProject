# CSE151A-GroupProject

# Milestone 2 Tasks:

### 1. Download data:

We manually downloaded multiple `listing.csv` files from [insideairbnb.com](https://insideairbnb.com/get-the-data/)
. We then renamed the each file according to its city and stored the file in `data` folder. Then, we ran [download_combine_data.ipynb](https://github.com/chengyuanmao819/CSE151A-GroupProject/blob/milestone2_Evaluate_Data_and_Plot_data/download_combine_data.ipynb) to combine all the listings into `all_data.csv` file

### 2. How do we process our data?
  - View [Data Dictionary: ](https://docs.google.com/spreadsheets/d/1iWCNJcSutYqpULSQHlNyGInUvHg2BoUGoNRIGa6Szc4/edit?usp=sharing)   
  - Suggest: drop `id`, `name`, `host_id`, `host_name` (this one might play a role. Catchy names or words might increase click, but how to analyze this?), `neighbourhood_group`, `neighbourhood` (we already had `city` column, `neighbourhood` is too specific), `license`
  - keep: 
    - `latitude`, `longtitude` (these 2 might be well correlated). 
    - `room_type` might need to be encoded to integer, for example, `Entire home/apt`:3, `Private room`:2...
    - `minimum_nights`
    - `number_of_reviews`
    - `last_review`, `number_of_reviews_ltm`: this reflects how active the listing is, how often and recent it got rent. Convert the date so that the most recent has higher integer values?
    - `reviews_per_month`
    - `calculated_host_listings_count`
    - `availability_365`
    - `city`: encode to int for float
  - target: `price`

### 3. Do values need to be cropped? normalized?

Looking at part 1.2, "Details about the Data Distributions," we can see that the data is extremely skewed due to anomalies or outliers. For example, the feature price has a 75th percentile of $271 per night, but the maximum value is up to $100,000 per night, which is quite unrealistic. Similarly, reviews_per_month has a 75th percentile of 2.26, while its maximum value is 88.95. These extremes make the pairplot squeezed, and we can't visually interpret the data. Our solution is to remove these anomalies and outliers using the [Interquartile Range](https://medium.com/analytics-vidhya/removing-outliers-understanding-how-and-what-behind-the-magic-18a78ab480ff). 

The features that we look to remove extremes from are `['price', 'number_of_reviews_ltm', 'minimum_nights', 'calculated_host_listings_count']`

For our prediction, we will try building models that either train on non-normalized or normalized data. Since all our features seem normally distributed, we'll try z-score standardization of the data.

For unsupervised machine learning tasks, we'll use normalized data, so that the component vectors are not affected by the different scales of the features and can provide a more accurate representation of the underlying structure. Normalizing the data ensures that each feature contributes equally to the analysis, preventing features with larger scales from dominating the results.

# Milestone 2 Tasks:



### 4. Link to Jupyter Notebook: [milestone2.ipynb](https://github.com/chengyuanmao819/CSE151A-GroupProject/blob/milestone2/milestone2.ipynb)
