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

- Precision: 0.7252747252747253
- Recall: 0.6342088404868674
- Fbeta: 0.6766917293233083

## Ethical Considerations

If used incorrectly, the predicted information can affect people.

## Caveats and Recommendations

The data is relatively old that it does not precisely reflect the current distribution.
