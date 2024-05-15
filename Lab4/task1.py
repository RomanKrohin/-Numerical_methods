import pandas as pd
import numpy as np
from scipy import stats

file_path = 'cars93.csv'
data = pd.read_csv(file_path)

data.head()

# 1. Анализ распределения цен и сравнение с нормальным распределением

shapiro_test_full = stats.shapiro(data['Price'])
print(shapiro_test_full)

# 2. Анализ различий в распределениях цен и мощности между автомобилями разных стран происхождения

grouped_by_origin = data.groupby('Origin')

prices_by_origin = [group['Price'].values for name, group in grouped_by_origin]
horsepower_by_origin = [group['Horsepower'].values for name, group in grouped_by_origin]

anova_prices_result = stats.f_oneway(*prices_by_origin)
anova_horsepower_result = stats.f_oneway(*horsepower_by_origin)

print(anova_prices_result) 
print(anova_horsepower_result)

# 3. Анализ зависимости цены от мощности с использованием линейной регрессии для полного датасета

regression_result = stats.linregress(data['Horsepower'], data['Price'])

print(regression_result)

