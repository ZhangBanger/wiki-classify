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
```

The system first tries to find the data in the directory you specified or download via API and parse.
Then, it will train on a variety of classifiers and explore hyperparameters.
While the system is training, you'll get to see what models it's trying out.
It will also report the best hyperparameters it found.

#### Dataset

The script automatically breaks apart your data into training and test (80/20 split) at the file level.
If you wish to reshuffle or add more categories, the script is not smart enough to figure this out.
You must delete the old data (X_train.txt, etc) and rerun, or point to a new directory.
If you don't do this and try to train anyway, you'll end up with 0 examples in your new categories.

#### Customizing

You can extend this list of models the system tries to fit by adding new modeling pipelines in `trainer.py`.
However, whatever underlying `Estimator` you use must implement the `predict_proba()` method.

### Try predictions

* Quote a wikipedia article title
* Point to the model used for prediction

```bash
$ python predict.py -m $(pwd)/model.pkl -t "Monstrous birth"
```
