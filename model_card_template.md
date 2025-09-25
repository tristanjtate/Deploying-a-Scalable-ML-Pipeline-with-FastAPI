# Model Card

For additional information, see the [Model Card paper](https://arxiv.org/pdf/1810.03993.pdf).

## Model Details
This model is a **RandomForestClassifier**, version 1.0 (created in 2025).  
It was developed to classify income levels based on demographic and work-related features.

## Intended Use
The model is intended for educational purposes within the Udacity MLOps project.  
Its specific purpose is to predict whether an individual's income is greater than \$50K or less than or equal to \$50K.  
It should not be used in production systems or for making financial, employment, or policy decisions.

## Training Data
The model was trained using the **Census Income Dataset** (Adult dataset) from the UCI Machine Learning Repository.  
Dataset link: [Census Income Data](https://archive.ics.uci.edu/ml/datasets/census+income).  
The dataset includes features such as age, occupation, marital status, education, race, sex, hours worked per week, and native country.  
Categorical features were one-hot encoded, and the target label (`salary`) was binarized into two classes: `<=50K` and `>50K`.

## Evaluation Data
The evaluation was conducted on a test set consisting of **20% of the original dataset**.  
The split was performed randomly with stratification to preserve the target label distribution.

## Metrics
The model’s performance was evaluated using **precision, recall, and F1 score**.  
On the held-out test set, the results were:  
- Precision: **0.7419**  
- Recall: **0.6384**  
- F1 Score: **0.6863**

## Ethical Considerations
This model was trained on U.S. Census data collected in the 1990s.  
As such, it may reflect social and economic biases present in that time period.  
Predictions may vary across demographic groups, and caution should be taken to avoid unfair or discriminatory uses.

## Caveats and Recommendations
- The model is based on historical data and may not reflect current economic conditions or workforce patterns.  
- Performance may differ across slices of the data (e.g., by gender, race, or occupation).  
- Future improvements could involve hyperparameter tuning, use of more recent datasets, or applying fairness-aware machine learning techniques.
