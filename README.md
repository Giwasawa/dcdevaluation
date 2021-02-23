# dcdevaluation (Decode - Evaluation package)
dcdevaluation is a package that optimizes the model evaluation process.
It was built based on the necessities of the data science team of a company named Decode, based on São Paulo - Brazil.
As it is in it's early stages, for now, it only supports binary classification models.

# Supported metrics
- KS
- ROC AUC
- F1-Score
- Precision
- Recall
- Accuracy
- "Bad rate graph"

# Requirements
- scikit-learn (0.23.2)
- Pandas (1.0.5)
- Matplotlib (3.3.0)
- Numpy(1.18.5)

# User installation
- Installing
```sh
pip install dcdevaluation
```
- Importing
```sh
from dcdevaluation import Evaluators
```
# Methods
### Init
To use the "Evaluators" class, instanciate it into a python object passing a list of probabilities (will be outputed by your model) and the true value of your data base (target feature used to train your model)
- E.g.:
``` sh
train_dataset = Evaluators(predicted_y, true_y)
```

### evaluate()
Attributes scores for all supported metrics(see above) to your select "dataset"
```sh
train_dataset.evaluate()
```
This method returns:
``` sh
train_dataset.ks
train_dataset.auc
train_dataset.f1
train_dataset.precision
train_dataset.recall
train_dataset.accuracy
```

### to_table()
Creates a pandas dataframe with all supported metrics
``` sh
train_dataset.to_table()
```
This method returns:
``` sh
# DataFrame with all supported metrics
train_dataset.metric_df
# Transposed DataFrame
train_dataset.t_metric_df
```
### split_rate_graph(bins)
Creates a graph showing the good or bad rate of your model

Has the attribute bins, which allows the user to change the desired number of splits (default = 10)

``` sh
train_dataset.split_rate_graph()
```

### find_cut(min, max)
Shows precision,recall and F1 score for 20 different cutting points.

Also has the option to select a range of cutting points (default: min = 0, max = 20)

``` sh
train_dataset.find_cut()
```

### ROC_curve(dataset)
Creates as graph showing de ROC curve and it's comparison to "the coin".

``` sh
train_dataset.ROC_auc()
```
 
