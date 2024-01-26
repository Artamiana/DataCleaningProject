import numpy as np
import pandas as pd

def find_outliers_iqr(data, feature, log_scale=False, left=1.5, right=1.5):
    """Функция по очистке данных от выбросов по методу межквартильного размаха (метод Тьюки). 
    iqr (верхнюю и нижнюю границы) можно менять как влево, так и вправо, также можно использовать 
    логарифмирование при необходимости. 

    Args:
        data (pandas.DataFrame): датасет, нуждающийся в очистке от выбросов
        feature (str): наименование столбца очищаемого датасета
        log_scale (bool, optional): Логарифмирование (необязательно, по умолчанию False). 
        left (float, optional): коэффициент для левой границы. По умолчанию выставлено на 1.5.
        right (float, optional): коэффициент для правой границы. По умолчанию выставлено на 1.5.

    Returns:
        outliers (pandas.DataFrame): наблюдения, попавшие в разряд выбросов
        cleaned (pandas.DataFrame): очищенные от выбросов данные
        """
    if log_scale:
        x = np.log(data[feature])
    else:
        x = data[feature]
    
    quartile_1, quartile_3 = x.quantile(0.25), x.quantile(0.75),
    iqr = quartile_3 - quartile_1
    lower_bound = quartile_1 - (iqr * left)
    upper_bound = quartile_3 + (iqr * right)
    outliers = data[(x < lower_bound) | (x > upper_bound)]
    cleaned = data[(x >= lower_bound) & (x <= upper_bound)]
    return outliers, cleaned


def find_outliers_z_score(data, feature, log_scale=False, left=3, right=3):
    """Функция по очистке данных от выбросов по методу z-отклонений (метод 3х сигм). 
    Количество сигм можно менять как влево, так и вправо, также можно использовать 
    логарифмирование при необходимости. 

    Args:
        data (pandas.DataFrame): датасет, нуждающийся в очистке от выбросов
        feature (str): наименование столбца очищаемого датасета
        log_scale (bool, optional): Логарифмирование (необязательно, по умолчанию False). 
        left (int, optional): число сигм (стандартных отклонений) влево. По умолчанию выставлено на 3.
        right (int, optional): число сигм (стандартных отклонений) вправо. По умолчанию выставлено на 3.

    Returns:
        outliers (pandas.DataFrame): наблюдения, попавшие в разряд выбросов
        cleaned (pandas.DataFrame): очищенные от выбросов данные
    """
    if log_scale:
        x = np.log(data[feature]+1)
    else:
        x = data[feature]
    mu = x.mean()
    sigma = x.std()
    lower_bound = mu - left * sigma
    upper_bound = mu + right * sigma
    outliers = data[(x < lower_bound) | (x > upper_bound)]
    cleaned = data[(x >= lower_bound) & (x <= upper_bound)]
    return outliers, cleaned
