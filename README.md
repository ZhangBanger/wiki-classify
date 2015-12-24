# Wiki Classify

Build a multi-class classifier for wikipedia articles 

## Instructions

### Add categories
Add the exact name of the category, complete with disambiguation text, to categories.txt

```bash
$ cat > categories.txt
 Organs_(anatomy)
^D
```

### Build classifier

* Select a directory to store or load your training data, as well as the path to your model
* Build your classifier

```bash
$ python build_classifier.py -m $(pwd) -d $(pwd)
2015-12-23 18:55:25,401 Collecting data for all categories
2015-12-23 18:55:25,402 Found data in /Users/andy/code/python/wiki-classify:
2015-12-23 18:55:25,412 Collected 928 examples across 7 categories
2015-12-23 18:55:25,418 Performing Randomized Search over MultinomialNB Parameters
Fitting 4 folds for each of 50 candidates, totalling 200 fits
...
2015-12-23 18:19:16,813 Pickled and saved model to /Users/andy/code/python/wiki-classify/model.pkl
```

The system first tries to find the data in the directory you specified or download via API and parse.
Then, it will train on a variety of classifiers and explore hyperparameters.
While the system is training, you'll get to see what models it's trying out.
It will also report the best hyperparameters it found.

#### Dataset

The script hits the Wikipedia API and then applies some predefined processing from the `gensim` package, as well as a few items I added myself.
It then partitions your data at the file level into training and test set, assuming an 80/20 split.
If you wish to reshuffle or add more categories, you must delete the old data (X_train.txt, etc) and rerun, or point to a new directory.
If you don't do this and try to train anyway, you'll end up with 0 examples in your new categories.

#### Customizing

You can extend this list of models the system tries to fit by adding new modeling pipelines in `trainer.py`.
However, any underlying `Estimator` you use must implement the `predict_proba()` method.

If you look at some past commits, you'll see attempted pipeline elements that didn't make the cut.
Some examples include SVMs with various kernels, bigrams, and decomposition steps like SVD, LDA, and NMF.
None of the methods resulted in improvement, some significantly worsened performance, and all incurred more training time by adding steps and introducing more hyperparameters to optimize.
For example, hinge loss SVMs were close to Logistic Regression in performance, but only the inefficient implementation with supra-quadratic running time supports `predict_proba()`. In any case, they're known to generate particularly bad "probability" estimates that require a lot of calibration.

### Try predictions

* Quote a wikipedia article title
* Point to the model used for prediction

```bash
$ python predict.py -m $(pwd) -t "Monstrous birth"
Tags for 'Monstrous birth': Congenital_disorders
Tag probabilities
Congenital_disorders -> 0.80
Infectious_diseases -> 0.25
Rare_diseases -> 0.21
Medical_devices -> 0.20
Organs_(anatomy) -> 0.19
Cancer -> 0.14
Machine_learning_algorithms -> 0.08
```

## Future Work

#### Labeled LDA

With the new multi-label formulation in mind, it's worth trying labeled LDA to see if any interesting results come about. See [this paper](http://www-nlp.stanford.edu/cmanning/papers/llda-emnlp09.pdf).
