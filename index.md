## Lab 3

### Questions

Q1. (T/F) An "ordinary least squares" (or OLS) model seeks to minimize the differences between your true and estimated dependent variable.

Answer: True. When minimizing the sum of the squared residuals, you are also minimizing x.

Q2. (Agree/Disagree) In a linear regression model, all feature must correlate with the noise in order to obtain a good fit.

Answer: Disagree. Noise is an independent error term and linear regression models can still have a decent fit without adjusting to all of the noise.

Q3. Write your own code to import L3Data.csv into python as a data frame. Then save the feature values  'days online','views','contributions','answers' into a matrix x and consider 'Grade' values as the dependent variable. If you separate the data into Train & Test with test_size=0.25 and random_state = 1234. If we use the features of x to build a multiple linear regression model for predicting y then the root mean square error on the test data is close to:

```markdown
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

df = pd.read_csv('L3Data.csv')
X = df[['days online','views','contributions','answers']].values
y = df['Grade'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=1234)

model = LinearRegression()
model.fit(X_train, y_train) # Always fit on train data
y_pred = model.predict(X_test) # Predict tested data

rmse = np.sqrt(mean_squared_error(y_test,y_pred))
print(rmse)
```
Answer: 8.3244

Q4. (T/F) In practice we determine the weights for linear regression with the "X_test" data.

Answer: False. We determine the weights for linear regression with the trained data.

Q5. (T/F) Polynomial regression is best suited for functional relationships that are non-linear in weights.

Answer: False. Even though Polynomial regression allows for a non-linear relationship between X and Y, this type of regression is still considered a linear regression in terms of weights.

Q6. (T/F) Linear regression, multiple linear regression, and polynomial regression can be all fit using LinearRegression() from the sklearn.linear_model module in Python.

Answer: True, these can all be fitted using LinearRegression() from the sklearn.linear_model module.

Q7. Write your own code to import L3Data.csv into python as a data frame. Then save the feature values  'days online','views','contributions','answers' into a matrix x and consider 'Grade' values as the dependent variable. If you separate the data into Train & Test with test_size=0.25 and random_state = 1234, then the number of observations we have in the Train data is:

```markdown
len(X_train)
```

Answer: 23

Q8. (T/F) The gradient descent method does not need any hyperparameters.

Answer: True, especially for the learning rate.

Q9. To create and display a figure using matplotlib.pyplot that has visual elements (scatterplot, labeling of the axes, display of grid), in what order would the below code need to be executed?

```markdown
import matplotlib.pyplot as plt

fig, ax = plt.subplots()
ax.scatter(X_test, y_test, color="black", label="Truth")
ax.scatter(X_test, lin_reg.predict(X_test), color="green", label="Linear")
ax.set_xlabel("Discussion Contributions")
ax.set_ylabel("Grade")

ax.grid(b=True,which='major', color ='grey', linestyle='-', alpha=0.8)
ax.grid(b=True,which='minor', color ='grey', linestyle='--', alpha=0.2)
ax.minorticks_on()
```

Q10. Which of the following forms is not  linear in the weights ?

Answer: Mathematical expression with the 4th power beta.
