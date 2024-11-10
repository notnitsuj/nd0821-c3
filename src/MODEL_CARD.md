# Model Card

## Model Details

The model is a `RandomForestClassifier` with the implementation from `scikit-learn` with default hyperparameters.

## Intended Use

The model is used to predict whether an employee's income exceeds $50k.

## Training Data

The model was trained using the [Census Income](https://archive.ics.uci.edu/dataset/20/census+income) data.

## Evaluation Data

The data was split into 80% for training and 20% for evaluation.

## Metrics

- Precision: 0.7387661843107388
- Recall: 0.6360655737704918
- Fbeta: 0.6835799859055673

## Ethical Considerations

If used incorrectly, the predicted information can affect people.

## Caveats and Recommendations

The data is relatively old that it does not precisely reflect the current distribution.
