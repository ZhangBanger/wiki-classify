# Wiki Classify

Build a multi-class classifier for wikipedia articles by defining existing categories 

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

While the classifier is training, you'll get to see what models it's trying out.

#### Customizing

You can extend this list by adding new modeling pipelines in `trainer.py`.
However, whatever underlying `Estimator` you use must implement the `predict_proba()` method.

### Try predictions

* Quote a wikipedia article title
* Point to the model used for prediction

```bash
$ python predict.py -m $(pwd)/model.pkl -t "Monstrous birth"
```
