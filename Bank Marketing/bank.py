import numpy as np
import pandas as pd

# Obtener data

url = 'bank-full.csv'
data = pd.read_csv(url)

# Tratamiento data

data.age.replace(np.nan, 41, inplace=True)
rangos = [0, 8, 15, 18, 25, 40, 60, 100]
nombres = ['1', '2', '3', '4', '5', '6', '7']
data.age = pd.cut(data.age, rangos, labels=nombres)
data.dropna(axis=0,how='any', inplace=True)
data.marital.replace(['married', 'single', 'divorced'], [0, 1, 2], inplace=True)
data.education.replace(['unknown', 'primary', 'secondary', 'tertiary'], [0, 1, 2, 3], inplace=True)
data.default.replace(['yes', 'no'], [0, 1], inplace=True)
data.housing.replace(['yes', 'no'], [0, 1], inplace=True)
data.loan.replace(['yes', 'no'], [0, 1], inplace=True)
data.contact.replace(['cellular', 'unknown', 'telephone'], [0, 1, 2], inplace=True)
data.poutcome.replace(['unknown', 'failure', 'other', 'success'], [0, 1, 2, 3], inplace=True)
data.y.replace(['yes', 'no'], [0, 1], inplace=True)

data.drop(['balance', 'duration', 'campaign', 'pdays', 'previous', 'job', 'day', 'month'], axis=1, inplace=True)

