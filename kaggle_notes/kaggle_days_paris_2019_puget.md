# Jean-Francois Puget

You are not supposed to share information outside of a team. Having more than one account is against the rules.

## Exploratory Data Analysis

Find interesting insights, even if the plot is not pretty. Most people show the obvious in nice colours.

2Sigma competition: you had to predict whether an apartment ad would raise interest (high, medium, low) given a description, location, picture, number of rooms, weekly price and other features. There was an important leak in this challenge. JFP plotted the time of the image file (upload time? JFP does not remember what is on the axes). The colors are not uniformly spread, but there are clear clusters.

Partial Dependency Plots can be useful: JFP shows an example of a modified version where the color is the target, the x axis is a feature index and the y axis is its frequency. Some important patterns emerged from this plot.

JFP does not see the point in plotting the correlation between features, but he does in plotting target vs. features.

It is very important to submit right after EDA. Don't start building a sophisticated pipeline. Submit to make sure that:

1. You understand the problem correctly.
2. Your code runs.

Then you start adding things with an incremental approach. JFP want to have a sound and scientific approach, where you iterate and test hypotheses. You hypothesize that a certain feature may help. You design an experiment, you validate it locally, you submit it, if it's successful good, if not, learn from it.

Don't do two things at the same time. Always test *one* thing at a time.

## Cross Validation Setting

The most important thing, both in Kaggle and in real life, is to make sure to have a reliable way to evaluate your model with your training data. If you don't, you can only submit and look at the result on the public leaderboard. This is an example of *LB probing*. LB probing can only work if:

1. The test data is really similar to the training data.
2. The test data is large enough.

In all other cases it is just gambling. JFP says that his private LB rank is always greater (I suppose he means better) than his public LB one. The correct way is through Cross Validation. You need to adjust the number of folds depending on the amount of data and the running times. The goal is that if your score in CV improves, you should see an improvement in the LB as well. Not necessarily by the same amount, but it should improve monotonically. If this works, you don't have to worry about the LB anymore, and you can run as many experiments as you want locally.

## Checking the CV score on the *training* set

Example where JFP has two models, with local CV score 1 > CV score 2 (higher is better). On the LB, however, it is the other way around. He also computed the score on the training data for each fold and took the average. Here are the numbers:

| Train score | CV score | LB score |
|-------------|----------|----------|
| 0.99004     | 0.98040  | 0.9675   |
| 0.98313     | 0.97937  | 0.9694   |

In this example the training score (this is the AUC) is very high, and there is only ~1% of the area *above* the ROC curve. On the other hand, this *doubles* when you go to the CV score. If the error doubles from training to CV, you are overfitting. In the second case the increase is much lower. This is why JFP always looks also at the training score, and if this improves too much, he knows that there is a high risk of overfitting. He also avoids features that provide a small CV improvement at the expense of a large training improvement.

## Feature Engineering: try a lot and fail fast

Great [slide deck](https://www.slideshare.net/HJvanVeen/feature-engineering-72376750) on feature engineering. There are a number of really useful techniques. If you are using XGBoost and LightGBM as the primary method, you should not worry about missing values and one-hot encoding. The *killer features*, however, never come from these tricks, but always from a deep understanding of the problem.

## HPO

A lot of people start with XGBoost and similar methods, and spend a lot of time tuning all the parameters. You should do a bit of tuning, but if you do it a lot, you are sure to overfit. He does it once or twice. Once after a couple of weeks when he has something reasonable and maybe once before submission, but that's it. Each time you do it, you overfit to the training data. Moreover, if you do HPO in the middle of a competition, you can no longer compare your results with the ones you had before. JFP tunes a small number of knobs:

1. Start with subsample 0.7. Leave other values to default.
2. Play with `min_child_weight`. If you use log-loss, this is the number of samples in each leave. Increase if the train/val gap is large.
3. Then tune `max_depth` or `number_of_leaves`.
4. Add regularization if LB score is way below CV.

## Ensembling

Until two years ago this was the main difference between grandmasters and amateurs. This is changing. JFP is convinced, and so is "BestFitting" in the Talking Data competition, that a great model is better than an ensemble. It is better to spend a lot of time finding a great model, and then you spend the last few weeks training new models on the dataset produced by this model (see below). Important things to keep in mind:

1. Use the same folds for all models.
2. Use out of fold predictions as feature for second level of models (stacking)
3. Keep in mind that the gap between CV and LB increases as there is some overfitting.

## 2Sigma Apartment Rental Competition

In this competition you have a description, the images, the location, thw weekly price, and you must predict the amount of interest in the ad (low, medium high). This is a very interesting competition for feature engineering. It combines image processing, NLP, geographical information etc. There was a leak, so you need to adjust for it.

JFP says that he did what you would normally do when looking at an ad. You want to understand whether the price is high or low w.r.t. the market. He trained a model to capture the market price. He took the description of all apartments, including those in the test set, and tried to predict the market price. JFP says that you can use all the data here without a problem. Then he uses as a feature the difference between the actual price and the price returned by the model. If the difference is negative, the actual price is below the market price.

Another feature has to do with how the streets in Manhattan are arranged. XGBoost splits data such that "if you plot two features, it will plot at right angles". An apartment can be a lot more expensive on one side of the street than on the other. He created a system of coordinates aligned with the streets, and this gave a boost.

## Talking Data Competition

350 million click data. Information was an ip, app, device, OS, channel, click time. All features were numbers, apart from the dates. When an application is downloaded, the `is_attributed` feature becomes 1. This was the target. The ip address is not unique (it is much more shared in China than in other countries). What features can you create here? If I'm interested in an application on my phone, I will click on it until I have downloaded it, but once I have, I will stop. The problem was to detect when people stop clicking. The killer feature was to compute the time to the next click for the same combination of ip, device, os, channel etc. Just thinking "how would I behave in this situation?" will help understanding the business problem and come up with new features.

## PLasticc: a research question

Time-series from a simulated telescope. Problem is to identify which time-series is a supernova. There were more classes in the test set than in the training set. JFP says he would like to see more challenges like this.

He used Gaussian Processes (code on GitHub). There are missing pieces in the time series. One of the given features was the "redshift". He plotted the redshift. Redshift alone turned out to be able to classify two classes alone. This should not be, due to the laws of Physics, and is actually a consequence of sampling bias. This was the most important feature, according to LightGBM, and JFP *removed* it. Same thing in Talking Channel. This removal did not improve the CV but did improve the LB. The 2nd solution undid Redshift. The 5th solution weighted train samples by inverse redshift frequency.

## Learn from Deep Learning

Data Augmentation is a great idea coming from Deep Learning. Why don't we do the same in XGBoost. All the top 5 teams in Plasticc did some form of data augmentation.

Another thing DL is good at is learning representation. Jahrer's winning solution in the Porto Seguro competition is a good example. JFP implemented a Factorization Machine in Keras for Talking Machines.

## Target Engineering

One thing people do nowadays is to change the target such that a custom metric becomes a classical metric. An example is a metric where taking the $\log \hat y$ returns a weighted RMSLE, which XGBoost can handle easily.

Another example from the web traffic competition: the metric was asymmetric (SMAPE), where low predictions are penalized more than the high ones. If you take the log, you get something close mean average ??

Also from Talking Data: for the same feature you could have repeated rows with different targets. This means that clicks were too close in time (within one second). He sorted in the training and the test data the timestamp and by the target, such that within each timestamp the low target comes before the high target. What happened was that while preparing the dataset, the test data was sorted, but the training set was not.