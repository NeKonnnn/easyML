import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import norm
from statsmodels.tsa.seasonal import STL
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller, acf, pacf
from utils import timing

class EDATimeSeries:
    def __init__(self, df, output_dir="DATA_OUT/graphics"):
        """
        Инициализация класса для анализа временных рядов.

        Параметры:
            df (pd.DataFrame): Входной DataFrame.
            output_dir (str): Директория для сохранения графиков.
        """
        self.df = df
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs("DATA_OUT", exist_ok=True)

    @timing
    def analyze_datetime_attributes(self, datetime_column):
        """
        Анализ временных атрибутов (год, месяц, день, час) с построением графиков.
        Возвращает словарь с DataFrame для каждого атрибута.
        """
        self.df[datetime_column] = pd.to_datetime(self.df[datetime_column])
        self.df['Год'] = self.df[datetime_column].dt.year
        self.df['Месяц'] = self.df[datetime_column].dt.month
        self.df['День'] = self.df[datetime_column].dt.day
        self.df['Час'] = self.df[datetime_column].dt.hour

        attributes = ['Год', 'Месяц', 'День', 'Час']
        summary_tables = {}

        for attr in attributes:
            data = self.df[attr].dropna()
            if data.nunique() <= 1:
                print(f"Пропущен график для '{attr}', так как значения одинаковы или отсутствует дисперсия.")
                continue

            plt.figure(figsize=(10, 6))
            sns.histplot(
                data, 
                kde=True, 
                bins=30, 
                color='royalblue', 
                edgecolor='black', 
                alpha=0.8, 
                line_kws={"color": "orange", "linewidth": 2}, 
                label="KDE (плотность)"
            )
            mean = data.mean()
            std_dev = data.std()
            x_range = np.linspace(data.min(), data.max(), 1000)
            ideal_pdf = norm.pdf(x_range, loc=mean, scale=std_dev)
            ideal_pdf_scaled = ideal_pdf * len(data) * (data.max() - data.min()) / 30
            plt.plot(
                x_range, 
                ideal_pdf_scaled, 
                color="red", 
                linestyle="--", 
                linewidth=2.5, 
                label="Идеальное распределение"
            )
            median = data.median()
            plt.axvline(median, color="blue", linestyle="--", linewidth=2, label=f"Медиана: {median:.2f}")
            plt.axvline(mean, color="green", linestyle="-.", linewidth=2, label=f"Среднее: {mean:.2f}")
            plt.title(f"Распределение по '{attr}'", fontsize=16)
            plt.xlabel(attr, fontsize=12)
            plt.ylabel("Частота", fontsize=12)
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            plt.legend(fontsize=10)
            file_path = os.path.join(self.output_dir, f"datetime_attribute_distribution_{attr}.jpg")
            plt.savefig(file_path, bbox_inches='tight')
            plt.show()
            print(f"График распределения по '{attr}' сохранён в: {file_path}")
            summary_table = data.value_counts().sort_index().reset_index()
            summary_table.columns = [attr, 'Частота']
            summary_tables[attr] = summary_table

        return summary_tables

    @timing
    def decompose_time_series(self, datetime_column, value_column):
        """
        Декомпозиция временного ряда с выводом графика и расчетом статистик.
        Возвращает таблицу статистик для тренда, сезонности и остатков.
        """
        self.df[datetime_column] = pd.to_datetime(self.df[datetime_column])
        self.df = self.df.sort_values(by=datetime_column)
        ts = self.df.set_index(datetime_column)[value_column]
        stl = STL(ts, period=12)
        result = stl.fit()
        fig = result.plot()
        fig.set_size_inches(10, 8)
        plt.suptitle(f"Декомпозиция временного ряда: {value_column}", fontsize=16)
        for ax in fig.axes:
            ax.tick_params(axis="x", rotation=45)
        file_path = os.path.join(self.output_dir, f"stl_decomposition_{value_column}.jpg")
        plt.savefig(file_path, bbox_inches='tight')
        plt.show()
        plt.close()
        components = {
            "Trend": result.trend,
            "Seasonal": result.seasonal,
            "Residual": result.resid
        }
        stats = {key: {
            "Среднее": comp.mean(),
            "Стандартное отклонение": comp.std(),
            "Минимум": comp.min(),
            "Максимум": comp.max()
        } for key, comp in components.items()}
        return pd.DataFrame(stats).T

    @timing
    def plot_autocorrelations(self, datetime_column, value_column, lags=50):
        """
        Построить графики автокорреляции (ACF) и частичной автокорреляции (PACF)
        и вернуть таблицу значений.
        """
        self.df[datetime_column] = pd.to_datetime(self.df[datetime_column])
        self.df = self.df.sort_values(by=datetime_column)
        ts = self.df.set_index(datetime_column)[value_column]
        acf_values = acf(ts, nlags=lags, fft=True)
        pacf_values = pacf(ts, nlags=lags)
        plt.figure(figsize=(12, 6))
        plot_acf(ts, lags=lags, title="ACF (Автокорреляция)")
        plt.savefig(os.path.join(self.output_dir, f"acf_{value_column}.jpg"), bbox_inches='tight')
        plt.show()
        plt.figure(figsize=(12, 6))
        plot_pacf(ts, lags=lags, title="PACF (Частичная автокорреляция)")
        plt.savefig(os.path.join(self.output_dir, f"pacf_{value_column}.jpg"), bbox_inches='tight')
        plt.show()
        acf_pacf_table = pd.DataFrame({
            "Лаг": range(len(acf_values)),
            "ACF (Автокорреляция)": acf_values,
            "PACF (Частичная автокорреляция)": pacf_values
        })
        return acf_pacf_table

    @timing
    def check_stationarity(self, datetime_column, value_column):
        """
        Проверка стационарности временного ряда с использованием теста Дики-Фуллера.
        Возвращает таблицу с результатами теста.
        """
        self.df[datetime_column] = pd.to_datetime(self.df[datetime_column])
        self.df = self.df.sort_values(by=datetime_column)
        ts = self.df.set_index(datetime_column)[value_column]
        result = adfuller(ts.dropna())
        stats = {
            "ADF Statistic": result[0],
            "p-value": result[1],
            "Critical Value (1%)": result[4]["1%"],
            "Critical Value (5%)": result[4]["5%"],
            "Critical Value (10%)": result[4]["10%"],
            "Stationary": result[1] < 0.05
        }
        return pd.DataFrame([stats])

    @timing
    def plot_seasonality_heatmap(self, datetime_column, value_column, freq='month'):
        """
        Построить тепловую карту сезонности и вернуть таблицу, использованную для её построения.
        """
        self.df[datetime_column] = pd.to_datetime(self.df[datetime_column])
        self.df['Year'] = self.df[datetime_column].dt.year
        if freq == 'month':
            self.df['Month'] = self.df[datetime_column].dt.month
            pivot = self.df.pivot_table(index='Year', columns='Month', values=value_column, aggfunc='mean')
            ylabel, xlabel = "Год", "Месяц"
        elif freq == 'day_of_week':
            self.df['DayOfWeek'] = self.df[datetime_column].dt.dayofweek
            pivot = self.df.pivot_table(index='Year', columns='DayOfWeek', values=value_column, aggfunc='mean')
            ylabel, xlabel = "Год", "День недели"
        elif freq == 'hour':
            self.df['Hour'] = self.df[datetime_column].dt.hour
            pivot = self.df.pivot_table(index='Year', columns='Hour', values=value_column, aggfunc='mean')
            ylabel, xlabel = "Год", "Час"
        else:
            raise ValueError("Неверное значение freq. Используйте 'month', 'day_of_week' или 'hour'.")
        plt.figure(figsize=(12, 8))
        sns.heatmap(pivot, cmap='viridis', annot=True, fmt=".1f", linewidths=0.5)
        plt.title(f"Тепловая карта сезонности: {value_column}", fontsize=14)
        plt.xlabel(xlabel, fontsize=12)
        plt.ylabel(ylabel, fontsize=12)
        file_path = os.path.join(self.output_dir, f"seasonality_heatmap_{value_column}.jpg")
        plt.savefig(file_path, bbox_inches='tight')
        plt.show()
        return pivot

    @timing
    def plot_time_series_with_table(self, datetime_column, value_column):
        """
        Построить график временного ряда и вернуть таблицу данных.
        """
        self.df[datetime_column] = pd.to_datetime(self.df[datetime_column])
        self.df = self.df.sort_values(by=datetime_column)
        table_data = self.df[[datetime_column, value_column]].reset_index(drop=True)
        plt.figure(figsize=(12, 6))
        plt.plot(self.df[datetime_column], self.df[value_column], marker="o", linestyle="-", color="blue")
        plt.title(f"Временной ряд: {value_column}", fontsize=16)
        plt.xlabel("Дата", fontsize=12)
        plt.ylabel("Значение", fontsize=12)
        plt.grid(True)
        file_path = os.path.join(self.output_dir, f"time_series_{value_column}.jpg")
        plt.savefig(file_path, bbox_inches="tight")
        plt.show()
        print(f"График временного ряда сохранён в: {file_path}")
        return table_data