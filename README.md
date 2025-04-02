# Stock Prediction using Linear Regression

## Overview
Stock price prediction using machine learning focuses on estimating the future value of company stocks and other financial assets traded on an exchange. This task is complex due to the influence of multiple factors, including market trends, investor sentiment, economic conditions, and unpredictable fluctuations, making stock prices highly dynamic and volatile.

This project implements the Linear Regression algorithm to forecast stock prices, utilizing Python and essential libraries such as pandas, NumPy, matplotlib, and scikit-learn (sklearn).

## Linear Regression
Linear Regression is a core supervised machine learning algorithm that models the relationship between a dependent variable (target) and one or more independent variables (predictors). The mathematical representation is:

Y = mx + c
Where:
- Y is the dependent variable (target).
- x is the independent variable (predictor).
- m represents the coefficient (slope) of the independent variable.
- c is the y-intercept.

The goal of the algorithm is to determine the best-fitting linear equation that describes the relationship between the input and output variables, enabling accurate predictions.

## Project Workflow
- **Data Splitting**: The dataset is divided into training and testing sets, with 80% to the training set and the remaining 20% to the testing set.
(`from sklearn.model_selection import train_test_split`
`x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42) `)
- **Model Training**: A Linear Regression model is trained on the training set to determine the relationship between input features and stock prices.
(`from sklearn.linear_model import LinearRegression`
`model=LinearRegression()`
`model.fit(x_train,y_train)`)
- **Testing and Predictions**: The trained model predicts stock prices using the test data.
(`y_pred=model.predict(x_test)`)
- **Accuracy Assessment**: The model's accuracy is assessed, with an accuracy above 99.5% indicating strong predictive performance.
(`print(f"Model R² Score:{regressor.score(X_test,y_test)}")`)

- **Visualization**: The predicted stock prices are visualized against actual prices to evaluate the model's performanc
(`import matplotlib.pyplot as plt`
`plt.plot(y_test,y_pred)`)


## Technologies Used
- **Python**: Core programming language for the project.
- **Pandas, NumPy**: Essential libraries for data manipulation and numerical operations.
- **Matplotlib**: Used for visualizing data through charts and graphs.
- **Scikit-learn (sklearn)**: Machine learning framework for model implementation and evaluation.

## Usage
1. Clone the repository.
2. Install the required dependencies: (`pip install pandas numpy matplotlib scikit-learn`).
3. Execute the Python script to run stock price prediction using Linear Regression.
4. Review the model's accuracy results after evaluation.

## Support or Contact
For any inquiries or support regarding the Stock Prediction using Linear Regression project, feel free to reach out at shoryaprajapat@jklu.edu.in.

Let me know if you want any further refinements! 🚀
