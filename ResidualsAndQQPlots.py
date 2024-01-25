# We're assuming that at this point our linear models have been fit. Please note that we typically build diagnostic plots for traditional linear models where there is an assumption of linearity between predictors and the outcome unlike GLMs.
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import statsmodels.api as sm
from scipy.stats import probplot

# Calculate residuals
residuals = y_test - y_pred

# Residual plot
plt.figure(figsize=(10, 6))
plt.scatter(y_pred, residuals)
plt.axhline(y=0, color='red', linestyle='--')
plt.xlabel("Predicted LOS")
plt.ylabel("Residuals")
plt.title("Residuals vs Predicted LOS")
plt.show()

# QQ plot of residuals to check for normality
plt.figure(figsize=(6, 6))
probplot(residuals, dist="norm", plot=plt)
plt.title("QQ Plot of Residuals")
plt.show()

# Histogram of residuals to check the distribution
plt.figure(figsize=(10, 6))
sns.histplot(residuals, kde=True, bins=50)
plt.xlabel("Residuals")
plt.title("Distribution of Residuals")
plt.show()
