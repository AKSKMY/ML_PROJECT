
# SVM Model Performance Summary

## Model Information
- **Algorithm**: Support Vector Machine (SVM)
- **Implementation**: scikit-learn
- **Date Trained**: 2025-03-06 03:57

## Regression Model (SVR)
- **Model Type**: SVR(C=0.1, kernel='linear')
- **Mean Absolute Error (MAE)**: 0.0485
- **Training Time**: 0.85 seconds

## Classification Model (SVC)
- **Model Type**: SVC(C=0.1, class_weight='balanced', kernel='linear', probability=True,
    random_state=42)
- **Accuracy**: 1.0000
- **Training Time**: 4.35 seconds

## Classification Report
```
              precision    recall  f1-score   support

           0       1.00      1.00      1.00        85
           1       1.00      1.00      1.00        33

    accuracy                           1.00       118
   macro avg       1.00      1.00      1.00       118
weighted avg       1.00      1.00      1.00       118

```

## Model Advantages
- Support Vector Machines excel at handling high-dimensional data like our text features
- Works well with both numerical and text data
- Effective at finding clear decision boundaries between delay severity classes

## Visualizations
- Confusion Matrix: see 'svm_confusion_matrix.png'
- Regression Results: see 'svm_regression_results.png'
- Model Comparisons: see 'model_mae_comparison.png' and 'model_accuracy_comparison.png'
