# 2025-Sales-Forecasting-Assignment

# Sales-Forecasting-Assignment

This is my solution to an example sales forecasting problem, using anonymized data from a real-world online retailer.

# Objective
Given daily sales for a set of products, category labels for those products, and information on whether a product is sold at a discount on a particular day, predict the sales of each product for the next week. In this README, I will guide the reader through my entire process of analyzing and solving the problem.

# Data Analysis and Preprocessing
This section is implemented in the "preprocess_data.ipynb" jupyter notebook.

## Basic statistics

```
Unique identifiers in train set: 2837
Unique identifiers in test set:  2163
Unique categories in train set:  36
Unique categories in test set:   34
Number of identifiers in test but not in train: 0

Min train date: 2024-05-01
Max train date: 2025-04-30
Min test date:  2025-05-01
Max test date:  2025-05-07

Number of identifiers in train with no category: 5
Number of identifiers in test with no category:  4
```

Two things to note here:

1) There are two categories of products that exist in the training dataset but not in the testing dataset. Given the fact that the category of a product is the only information we have that relates products to each other (lacking other attributes such as brand, price, product type, etc.), we can safely remove the identifiers belonging to these extra categories, since they will not offer us any useful information about the products that we want to predict sales for.
2) A few products have no category information. Lacking any other method to relate them to similar products (other than perhaps selling patterns), I chose to assign them to an additional category called "unknown".

## Missing entries in our training dataset

A quick glance at the training dataset is enough to notice that the majority of products have missing entries for certain dates that lie between their oldest and their most recent dates with sales recorded. The average percentage of missing days for a product is 44.42%, which is way too large to be caused by errors in data entry. An obvious answer is that days with no sales do no appear in our training data, yet there exist a lot of rows with n_sold=0. To determine the cause of the missing entries, I plot both the availability of sales data and the number of sales for 5 random products.

<img width="1789" height="862" alt="missing_entries" src="https://github.com/user-attachments/assets/bd3f1d1a-718c-433f-9295-51af29bd0c39" />

Looking at the two plots side by side, we can see that the products with lots of missing dates are the products whose daily sales are close to zero, while products that sell well above zero items have nearly no missing dates. We can thus conclude that a missing date in the training dataframe corresponds to zero sales for a product that day, although it is not clear why entries with zero sales exist.

Since the missing entries represent almost half of our training dataset, I fill the missing entries with n_sold=0 and is_discounted=False.

## Addition of date features and aggregation to weekly sales

In an attempt to find any seasonal or product-lifetime related features, I add the following information to the training dataset: "year", "month", "week" (starting from 2024-05-01 as week 1, and increasing from there, without resetting for each new year), "weekday" and "is_holiday". "is_holiday" is true if the date is an official public holiday (red day) in Sweden. I aggregate the results on a weekly level because the objective is to predict total sales for each product for the entire next week.

To get a rough idea of what numbers to expect, I plot the average number of weekly sales per category, along with the number of products per category.

<img width="1389" height="790" alt="categories_sales_and_num_products" src="https://github.com/user-attachments/assets/d858637d-ef85-4e11-8667-31bd1d190511" />

As we can see, some categories with a small amount of products belonging to them have a ton of sales, while others with more products sell very little. We can therefore expect a lot of products to sell very few items daily, with lots of days with zero items sold, while a select few products dominate with huge amounts of daily sales.

## Evolution of sales over product lifetime

Next I examine if there is any trend between a product's lifespan (i.e. how many weeks since it first started being sold) and the amount of items sold. As always, I check the results aggregated to categories.

<img width="1185" height="590" alt="weeks_since_launch" src="https://github.com/user-attachments/assets/0f434e32-a3cd-4625-8f8a-5f143618f97b" />

There is no clear pattern here. One category with lots of sales shows an obvious trend, but considering our training data is confined to a single calendar year, it is likely it is a seasonal trend rather than a product-lifespan one.

## Sales per week (seasonal trend)

To detect any such seasonal trends, I plot the average weekly sales per category.

<img width="1386" height="690" alt="weekly" src="https://github.com/user-attachments/assets/47e5dac2-dbf8-495a-acc7-b1ed6694efd6" />

As expected, it's very similar to the previous plot, due our definition of "week 1" as the starting week in our training set. (It's slightly different due to some products starting to get sold at a later date.) One of the categories shows clear signs of seasonality. Potentially it represents medicine products, whose sales presumably spike during the "cold" half of the year.

## Effects of discounts on sales

Next, I examine the effects of discounts on the sales. Weeks that have at least one day where the product is sold at a discount are marked with "has_discount"=True.

<img width="1187" height="590" alt="discounts" src="https://github.com/user-attachments/assets/a619b866-4b27-4ca0-8577-3016fd1697ad" />

Predictably, most products experience a large surge in sales when discounted, although this relative change may appear large due to small absolute numbers.

Discounts present a problem: We don't have information about discounts on the test set. Looking for an obvious sign of seasonality, I check the distribution per week (again starting from 2024-05-01 as week 1). 

<img width="998" height="545" alt="discount_seasonality" src="https://github.com/user-attachments/assets/db6a2b9e-64e0-40f9-9be5-f84523e76b70" />

Although there are some very clear spikes, there is no obvious pattern on when a discount happens. I check whether discounts tend to occur during holidays, on both a weekly and a daily level:

```
Correlation between discounts and holidays on a weekly level: -0.09
Correlation between discounts and holidays on a daily level:  -0.03
```

Unfortunately there is no correlation between the two features.

## Effects of holidays on sales

Similarly, I examine the effects of holidays on the sales.

<img width="1187" height="590" alt="holidays" src="https://github.com/user-attachments/assets/ab38a6d9-2e8e-49c5-84a1-ab8fed46373d" />

For most categories, there is no significant change. There are, however, a few that show a significant change, positive or negative. Since Apotea is an online pharmacy, most of its products should be relatively inelastic in terms of demand. A few may be highly seasonal (like sunscreen) which may correlate with certain holidays (Midsommar, National Day, etc.)

## Days Since Last Discount / Holiday

Here I plot the average weekly sales for each number of weeks since last discount/holiday. (The spikes at ~52 are manually placed for new products who have never had a discount/holiday in their selling history.)

<img width="1390" height="590" alt="output" src="https://github.com/user-attachments/assets/022b84ac-46bd-4bbb-989b-3ac93107729f" />

Holidays don't seem to have any noticeable effect on sales, while discounts seem to have a strong delayed effect. This makes some intuitive sense (products get promoted and as a result get bought more later) although it somewhat conflicts with our previous finding that products of most categories sell more during the weeks they are on sale. Huge categories skewing numbers might be the cause here. 

# Choosing a Model

Based on the processed data, I decided to use a tree-based regression model, using the LightGBM library. The reasons being:

1) We only have one year of training data available and there is no obvious seasonality to our features. Thus, forecasting models such as Prophet [are not recommended](https://facebook.github.io/prophet/).
2) Most of our features are categorical.
3) Tree-based models are good at handling non-linear relationships, which is good since we didn't find many strong correlations between features and sales.
4) They can capture interactions between features (such as discounts and holidays, which may be correlated in some way).
5) They are easy to train and provide feature importance out of the box.

# Model Design and Training

This section is implemented in the "train_lightgbm_model.ipynb" jupyter notebook.

## Feature Engineering

As is typical in time-series prediction, I add rolling features with various window sizes to my data. The rolling window size varies from 1 to 5 weeks, and the function applied is mean (rolling average) and standard deviation. These extra features should offer us local (temporally) information that is very useful for the prediction of the next step in a time-series.

I also add:

* Category-level features (`category, category_mean_sales, category_std_sales`)
* Product-level features (`identifier_mean_sales, identifier_max_sales, identifier_std_sales`)

In addition, these existing features from my processed training set are used: `category, week, contains_holiday, weeks_since_launch, weeks_since_last_discount, weeks_since_last_holiday`

## Creating the Training and Validation Sets

My goal is to predict the next week of sales, given historical sales so far. For that reason, I assume that we are working towards a model that gets updated and runs weekly. As such, I split my training set containing 52 weeks into weeks 1-51 for the training set and week 52 for the validation set.

## Use of log1p for the predicted values

In our data, there is great discrepancy in the number of sales between products. Even though tree-based methods are relatively robust to outliers, the huge number of sales of some products can penalize our model too much for mistakes during training, especially when using metrics like root mean squared error (rmse). For that reason, I choose to filter all model predictions (and ground truth labels) through the natural logarithm, thus somewhat constraining the high number of sales to more reasonable numbers. Care needs to be taken to convert the predictions and labels back to their true values before presenting the results.

I also choose to use the symmetric mean absolute percentage error (smape) metric, which measures the relative difference between predicted and actual values, assigning equal weight to both overestimating and underestimating the actual value. (Unlike mean absolute percentage error, where overestimation errors are numerically unbounded.)

## LGBM Hyperparameter Tuning

I use Optuna to perform a wide search of hyperparameter values, using the following code:

    
    params = {
        "objective": "regression",
        "boosting_type": "gbdt",
        "num_leaves": trial.suggest_int("num_leaves", 10, 300),
        "learning_rate": trial.suggest_float("learning_rate", 0.0005, 0.3, log=True),
        "feature_fraction": trial.suggest_float("feature_fraction", 0.3, 1.0),
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.3, 1.0),
        "bagging_freq": trial.suggest_int("bagging_freq", 1, 10),
        "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 5, 100),
        "lambda_l1": trial.suggest_float("lambda_l1", 0.0, 10.0),
        "lambda_l2": trial.suggest_float("lambda_l2", 0.0, 10.0),
        "verbose": -1,
        "feature_pre_filter": False,
    }


## Feature importance

The LGBM model provide us with calculated feature importance statistics after training. The results are presented below.

These plots confirm our intuition about certain features being useful (average sales per product, weeks since last discount, sales of most recent past week).

## Creation of the Final Model and Predictions

Finally, using the hyperparameters obtained through the previous training and validation steps, I train one final model, using all the available training data as my training set and the test data as my test set. Since there is no information given about discounts for the test set, I assume none exist.

The results, saved in `lightgbm_predictions.parquet`, are thus:

```
Average predicted sales: 6.37
Median predicted sales: 1.0
Max predicted sales: 3066
Min predicted sales: 0
```

Some notes:

1) The average prediction is very low. That makes sense, considering that most products sell a very small amount of items per day, or even none.
2) Max predicted sales for a week being 3066 is very low, considering what we can see in the training dataset. The product with identifier `049903176d`, for example, sold 15175 items the week before, so we can be certain that the actual number of sales for it for the test period is much higher than our 3066 number. This showcases a weakness of our model, and the need to model certain "evergreen" products that behave differently from the rest separately. For this assignment, I consider the work so far to be enough, but in the future a different approach for certain categories of products could be used.
3) Depending on what kind of prediction error we want to minimize (such as reducing the absolute amount of unsold stock vs minimizing percentage error in our sales forecasting), the evaluation metric for my model could be adjusted.
