# Adjusted-MAE-loss-function
# DEPRECATED - to be updated
Adjusted MAE loss function is a custom loss function for Pytorch that integrates a penalty for the difference in sign between the true y and the predicted y. 

Assuming I want to assess an incease or a decrease of a time series (fraffic, electricity consumption, asset price, ...), the sutom loss function penalizes the loss when the predicted y has a different signthan the the true y.

The function computes the product of y and yhat: 

=> If the product has a positive sign, y and yhat are of the same sign; 

=> If the product is negative yhat is of the opposite sign

Adjusted loss = loss * adjustment, where "adjustment = exp(-y * yhat / factor)
and 'factor = mean(abs(y))**2


The adjusted loss function significantly improves the convergence towards y with yhat of the same sign as y, compared to MAE loss function.

 - MAE Loss function is called with nn.L1Loss()
 - Adjusted MAE Loss function with AdjMAELoss()

