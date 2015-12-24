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
$ python build_classifier.py -m $(pwd)/model.pkl -d $(pwd)
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
It then breaks apart your data into training and test set, assuming an 80/20 split, at the file level.
If you wish to reshuffle or add more categories, the script is not smart enough to figure this out.
You must delete the old data (X_train.txt, etc) and rerun, or point to a new directory.
If you don't do this and try to train anyway, you'll end up with 0 examples in your new categories.

#### Customizing

You can extend this list of models the system tries to fit by adding new modeling pipelines in `trainer.py`.
However, whatever underlying `Estimator` you use must implement the `predict_proba()` method.

If you look at my other branches (like `svd`) and a few past commits, you'll see other pipeline elements that were attempted but didn't make the cut.
Some examples include SVMs with various kernels, bigrams, and decomposition steps like SVD, LDA, and NMF.
None of the methods offered any improvement, some significantly worsened it, and all of them incurred more training time by adding costly steps and introducing more hyperparameters to optimize.
For example, hinge-loss SVMs were close to Logistic Regression in performance, but only the inefficient implementation with supra-quadratic running time supports `predict_proba()`. In any case, they're known to generate particularly bad "probability" estimates that require a lot of calibration.

### Try predictions

* Quote a wikipedia article title
* Point to the model used for prediction

```bash
$ python predict.py -m $(pwd)/model.pkl -t "Monstrous birth"
Best prediction for 'Monstrous birth': Congenital_disorders
Prediction distribution
Congenital_disorders -> 0.31
Infectious_diseases -> 0.16
Medical_devices -> 0.16
Rare_diseases -> 0.14
Organs_(anatomy) -> 0.10
Cancer -> 0.10
Machine_learning_algorithms -> 0.05
```
