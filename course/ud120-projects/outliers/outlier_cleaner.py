#!/usr/bin/python


def outlierCleaner(predictions, ages, net_worths):
    """
        clean away the 10% of points that have the largest
        residual errors (different between the prediction
        and the actual net worth)

        return a list of tuples named cleaned_data where 
        each tuple is of the form (age, net_worth, error)
    """
    
    cleaned_data = []

    ### your code goes here
    residuals = abs(predictions - net_worths)
    x = zip(ages, net_worths, residuals)
    n = len(x)
    x.sort(key = lambda item : item[2])
    #cleaned_data = x[0:round(n * 0.9)] 
    cleaned_data = x[0:int(n * 0.9)] 
    
    return cleaned_data

