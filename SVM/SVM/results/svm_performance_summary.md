
# SVM Model for Motorcycle Price Prediction

## Model Overview
- **Algorithm**: Support Vector Machine (SVR)
- **Implementation**: scikit-learn
- **Kernel**: rbf
- **C (Regularization)**: 1000
- **Gamma**: 0.1
- **Epsilon**: 0.2

## Performance Metrics
- **Mean Absolute Error (MAE)**: $4243.69
- **Root Mean Squared Error (RMSE)**: $7564.30
- **R² Score**: 0.6411

## Feature Summary
The model was trained using the following features:
Brand, Model, Engine_Capacity, Registration_Date, COE_Expiry_Date, Mileage, No_of_owners, Category

## Visualization Summary
- **Actual vs Predicted**: Shows the relationship between actual and predicted motorcycle prices
- **Error Distribution**: Shows the distribution of prediction errors
- **Residual Plot**: Shows the relationship between predictions and their errors

## Model Advantages
- SVM can capture non-linear relationships in the data
- Robust to outliers when properly tuned
- Works well with both categorical (encoded) and numerical features

## Recommendations
- Consider feature engineering to capture more complex relationships
- Experiment with different kernel functions for potentially better performance
- Regularly update the model as new data becomes available
