import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
scale = StandardScaler()

df = pd.read_csv('kerala.csv')
df.head(5)

list_x = np.array(df["CO2 emission (Tons)"].tail(100).values)
list_y = np.array(df["Year"].tail(100).values)

print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
labels_x = []
labels_x = list_x.reshape((-1, 1))
for i in range(100):
    labels_x[i] = list_x[i]/100000
labels_y = list_y



model = LinearRegression().fit(labels_x, labels_y)
r_sq = model.score(labels_x, labels_y)
print('coefficient of determination:', r_sq)
print('intercept:', model.intercept_)
print('slope:', model.coef_)

y_pred = model.predict(labels_x)
print('Predicted response:', y_pred, sep='\n')

# Create a linear regression model based on the positioning of the data and Intercept, and predict a Best Fit:
model = LinearRegression(fit_intercept=True)
model.fit(labels_x, labels_y)

xfit = np.linspace(0, 100, 1000)  # Generate 1000 points from 0 to 10 (adjust range if needed)
xfit = np.arange(8000).reshape((-1,1))
yfit = model.predict(xfit.reshape((-1, 1)))  # Reshape xfit to have (n_samples, n_features)

# Plot the estimated linear regression line with matplotlib:


plt.scatter(labels_x, labels_y, color='hotpink')
plt.plot(xfit, yfit)
plt.xlabel('CO2 emission (Tons)')
plt.ylabel('Year')
plt.title('Linear Regression Fit')
plt.show()
