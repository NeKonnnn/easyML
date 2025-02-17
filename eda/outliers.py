import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import zscore
from utils import timing

class EDAOutliers:
    def __init__(self, df, output_dir="DATA_OUT/graphics"):
        """
        Инициализация класса для обнаружения выбросов.

        Параметры:
            df (pd.DataFrame): Входной DataFrame.
            output_dir (str): Директория для сохранения графиков.
        """
        self.df = df
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs("DATA_OUT", exist_ok=True)

    @timing
    def detect_outliers_iqr(self, numeric_columns):
        """
        Построить boxplot для числовых переменных, определить выбросы на основе IQR и сохранить графики.
        Возвращает сводную таблицу с информацией по выбросам.
        """
        outlier_summary = []
        for column in numeric_columns:
            Q1 = self.df[column].quantile(0.25)
            Q3 = self.df[column].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outliers = self.df[(self.df[column] < lower_bound) | (self.df[column] > upper_bound)][column].tolist()
            outlier_summary.append({
                'Признак': column,
                'Выбросы': outliers
            })
            plt.figure(figsize=(8, 4))
            sns.boxplot(x=self.df[column], palette="viridis")
            plt.title(f"Ящик с усами: {column}")
            plt.xlabel(column)
            file_path = os.path.join(self.output_dir, f"boxplot_{column}.jpg")
            plt.savefig(file_path, bbox_inches='tight')
            plt.show()
            plt.close()
        return pd.DataFrame(outlier_summary)

    @timing
    def detect_outliers_zscore(self, numeric_columns, threshold=3):
        """
        Найти выбросы на основе Z-score для числовых переменных и построить графики.
        Возвращает сводную таблицу с информацией по выбросам.
        """
        outlier_summary = []
        for column in numeric_columns:
            non_na_data = self.df[column].dropna()
            z_scores = zscore(non_na_data)
            outliers = non_na_data[(z_scores > threshold) | (z_scores < -threshold)].tolist()
            outlier_summary.append({
                'Признак': column,
                'Выбросы': outliers
            })
            plt.figure(figsize=(10, 6))
            sns.boxplot(x=self.df[column], palette="viridis")
            plt.title(f"Выбросы на основе Z-score: {column}")
            plt.xlabel(column)
            file_path = os.path.join(self.output_dir, f"zscore_boxplot_{column}.jpg")
            plt.savefig(file_path, bbox_inches='tight')
            plt.show()
            plt.close()
        return pd.DataFrame(outlier_summary)