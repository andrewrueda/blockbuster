# Andrew Rueda
# Predicting a Blockbuster

---
## Introduction
Having a background being in Film, I know that a lot of predictive insight can be drawn from certain meta-attributes of a movie: this goes beyond
easy associations like certain keywords and genres (although film genre is a pretty blurry and interesting construct in its own right),
but a movie is essentially an amalgamation of a bunch of different tropes, and since the industry tends to constantly copy itself, there are patterns that tend to merge, whether by design or osmosis.

For this project, I used a movie's synopsis (as well as other features in the metadata of the movie) to predict its global box office revenue.

My hypothesis was that movies with big budgets and wide releases are marketed using different language than indie films, and that even using short synopses, a model could pick up the slight differences in plot between a Blockbuster and a much lower budget film.

------
## Data
My initial dataset was from Kaggle: [Link](https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata)

It's scraped from the website TMDB, and it includes 5,000 movies with a **synopsis** and **revenue**, as well as a lot of detailed features, such as genre, runtime, release date,
cast/character info, etc.

|title|overview|release_date|revenue|runtime|tagline|
|---|---|---|---|---|---|
|Avatar|In the 22nd century, a paraplegic Marine is dispatched to the moon Pandora on a unique mission, but becomes torn between following orders and protecting an alien civilization.|2009-12-10|2787965087|162|Enter the world of Pandora.|
|Pirates of the Caribbean: At World's End|Captain Barbossa, long believed to be dead, has come back to life and is headed to the edge of the Earth with Will Turner and Elizabeth Swann. But nothing is quite as it seems.|2007-05-19|961000000|169|At the end of the world, the adventure begins.|


I split the data randomly into 80/10/10 for train/dev/test. My first and main issue was that, after filtering out movies with missing revenue data, there were only ~3,400 rows left to work with.

The upside was the richness in columns. My main feature for the model was BoW vectors from the synopsis. The data also came with the official Tagline and a list of keywords associated with the movie, so I included those words as part of the synopsis.

I also wanted to incorporate more of the movie's data to predict its revenue. Using the movie's release date, I added features for whether the movie was released in Q1, Q2, Q3, or Q4, and additionally, I included the movie's runtime as another feature.

Lastly, I did a spot check and observed that high-revenue movies tended to have more character names in the synopsis. So, as a final feature, I added an integer representing how many times a named character was mentioned in the synopsis, using a regex on the credits data.

The next step of course was picking the classes. I wanted to stick with Binary Classification, so I chose to predict whether a movie made OVER $100,000,00 in revenue, or UNDER.

I represented all of the features as a 2D array of ints, and the label as a string.

---------
## Modeling
My baseline results were interesting to start. After going through a lot of work extracting all of the above features, I felt that experimenting with different models was pretty easy/quick.

I experimented with Random Forest, Logistic Regression, and SVM models.
Here are my first round of Dev results:

|Model|HyperParameters|Accuracy|Over F1|Under F1|
|---|---|---|---|---|
|Random Forest|Default|72|46|81|
|Logistic Regression|Default|76|66|81|
|SVM|Default|65|20|78|

To my surprise, I realized that the F1 score for finding movies Under $100,000,000 was much better than Over. My intution was that Blockbusters were more homogenous in their attributes, the model was picking up something strong from the other side of the spectrum.
I did note that the support was also much greater for Under $100,000,000, so I experimented with changing the threshold to $50,000,000, as the data would be more even:

|Model|HyperParameters|Accuracy|Over F1|Under F1|
|---|---|---|---|---|
|Random Forest|Default|75|77|72|
|Logistic Regression|Default|72|72|73|
|SVM|Default|68|67|69|

----------

To be honest, I was pretty happy with these early numbers, especially when compared to the .29 F1 score I was getting for the Oscars project. However, before really getting into the Hyperparameters, I felt that the number of rows was insufficient.
I tried utilizing cross-validation, but ran into many problems and Timeout errors. Luckily, I took another look at Kaggle and found what is basically an update to this dataset, but with more rows:

[Link](https://www.kaggle.com/datasets/rounakbanik/the-movies-dataset?select=movies_metadata.csv)

I was quickly disappointed that, among these 45k rows, only about 7k had revenue data, but that was still an upgrade. I updated the dataset and got these results (Over/Under $50M):

|Model|HyperParameters|Accuracy|Over F1|Under F1|
|---|---|---|---|---|
|Random Forest|Default|80|48|87|
|Logistic Regression|Default|79|59|86|
|SVM|Default|74|18|84|

With this larger dataset, the support skewed back to about a 2:1 ratio of Under labels vs Over, and, looking at Precision and Recall, all versions were clearly over-tagging with the Under label.
Accuracy was up, but probably due to this skew.

-------
## Tuning

Next, I wanted to test out Hyperparameters. This is where I narrowed it down to Random Forest or Logistic Regression, and so played around with the most important Hyperparameters:

|Model|# of estimators / C|Accuracy|Over F1|Under F1|
|---|---|---|---|---|
|Random Forest|200|80|48|87|
|**Random Forest**|**600**|**80**|**48**|**88**|
|Random Forest|1000|80|47|87|
|Random Forest|1600|79|47|87|
|Random Forest|2000|79|47|87|
|Logistic Regression|.001|76|36|85|
|Logistic Regression|.01|80|56|87|
|**Logistic Regression**|**.1**|**81**|**61**|**87**|
|Logistic Regression|1|79|59|86|
|**Logistic Regression**|**10**|**79**|**72**|**78**|
|Logistic Regression|100|79|59|86|

----

## Testing
Of all these configurations on the Dev set, I highlighted 3 to run on the Test Set. They were generally the best scores (though all were quite even), and I was particularly interested in Logistic Regression, C=10, as that actually gave a decent F1 Score on the Over label.

|Model|# of estimators / C|Accuracy|Over F1|Under F1|
|---|---|---|---|---|
|Random Forest|600|76|41|85|
|Logistic Regression|.1|80|60|86|
|Logistic Regression|10|78|59|85|


|Model|# of estimators / C|Over Precision|Over Recall|Under Precision|Under Recall|Macro Avg|Weighted Avg|
|---|---|---|---|---|---|---|---|
|Random Forest|600|80|31|77|97|63|72|
|Logistic Regression|.1|72|52|82|92|73|78|
|Logistic Regression|10|66|54|82|88|72|77|

Well, Logistic Regression C=10 kind of flopped in the test set, in terms of having a good F1 score for the Over label. But overall, the accuracy staying at ~80% was a positive.
My last effort was to pick the best overall-performing configuration (Logistic Regression, C=.1) and adjust the class weights.

---
## Last Step

### Development:
|Model|C|Weights: Over/Under|Accuracy|Over F1|Under F1|
|---|---|---|---|---|---|
|Logistic Regression|.1|.75/.25|80|60|86|
|Logistic Regression|.1|.66/.33|80|63|86|
|**Logistic Regression**|**.1**|**.6/.4**|**81**|**63**|**87**|

### Test:
|Model|C|Weights: Over/Under|Accuracy|Over F1|Under F1|
|---|---|---|---|---|---|
|Logistic Regression|.1|.6/.4|80|63|86|


|Over Precision|Over Recall|Under Precision|Under Recall|Macro Avg|Weighted Avg|
|---|---|---|---|---|---|
|70|57|83|90|75|79|
-----
## Discussion and Conclusion

One crucial thing I should note is that I did run experiments of my model using only the synopsis BoW as features. It performed decently well, but slightly worse than above. This is a good indication that this model is being driven by the BoW features, and its score is being supplemented by the other features.
This is a relief, as it would be bad if it turned out that the Runtime scalar was doing all of the work, and my actual NLP features were useless.

I'm surprised that this model is better at classifying movies Under $50M in revenue than Over $50M, **despite** the skew of the updated dataset. My initial idea was that the Harry Potters and Marvels of the world would be easy to identify by their various features, but that it didn't really happen with any of these configurations.

The three types of Models I used were relatively even in performance, but I'm not surprised that Logistic Regression was the best choice.

Overall, I'm pretty happy with the final scores, especially given the fact that the correlation between the features and the label was a little speculative on my part. I could imagine a scenario where the accuracy was just 50% no matter what I did, so it's cool to see a decent performance, even though it's **worst** at successfully identifying a Blockbuster.
