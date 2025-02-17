import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import norm, kurtosis, skew, gaussian_kde
from utils import timing

class EDAVisualization:
    def __init__(self, df, output_dir="DATA_OUT/graphics"):
        """
        Инициализация класса для визуализации данных.

        Параметры:
            df (pd.DataFrame): Входной DataFrame.
            output_dir (str): Директория для сохранения графиков.
        """
        self.df = df
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs("DATA_OUT", exist_ok=True)

    @timing
    def plot_target_distribution(self, target_column):
        """
        Построить график распределения целевой переменной и сохранить его в файл.
        Возвращает сводную таблицу распределения.
        """
        target_counts = self.df[target_column].value_counts()
        ax = sns.barplot(x=target_counts.index, y=target_counts.values, palette="viridis")
        plt.title("Распределение целевой переменной")
        plt.xlabel("Значение целевой переменной")
        plt.ylabel("Количество")
        for p in ax.patches:
            height = p.get_height()
            ax.annotate(f'{int(height)}', (p.get_x() + p.get_width() / 2., height), ha='center', va='bottom')
        file_path = os.path.join(self.output_dir, f"target_distribution_{target_column}.jpg")
        plt.savefig(file_path, bbox_inches='tight')
        plt.show()
        plt.close()

        target_summary = pd.DataFrame({
            'Значение': target_counts.index,
            'Количество': target_counts.values
        })
        return target_summary

    @timing
    def plot_categorical_distributions(self, categorical_columns):
        """
        Построить графики распределения для категориальных переменных и сохранить их.
        Возвращает объединённую сводную таблицу.
        """
        category_summary = []

        for column in categorical_columns:
            non_na_data = self.df[column].dropna()
            if non_na_data.empty:
                print(f"Пропущен график для столбца {column}, так как он пустой после удаления пропусков.")
                continue
            try:
                parsed_dates = pd.to_datetime(non_na_data, format='%d.%m.%Y', errors='coerce')
                if parsed_dates.notna().mean() > 0.8:
                    print(f"Пропущен график для столбца {column}, так как он содержит данные формата даты.")
                    continue
            except Exception as e:
                print(f"Ошибка при проверке формата даты для столбца {column}: {e}")
                continue
            if pd.api.types.is_numeric_dtype(non_na_data):
                print(f"Пропущен график для столбца {column}, так как он содержит числовые данные.")
                continue

            unique_count = non_na_data.nunique()
            width = min(max(8, unique_count * 0.5), 20)
            rotation = 0 if unique_count <= 5 else 30 if unique_count <= 10 else 60
            height = 6

            plt.figure(figsize=(width, height))
            ax = sns.countplot(x=non_na_data.astype(str), palette="viridis")
            plt.title(f"Распределение категориальной переменной: {column}")
            plt.xlabel(column)
            plt.ylabel("Количество")
            plt.xticks(rotation=rotation)
            for p in ax.patches:
                height_val = p.get_height()
                ax.annotate(f'{int(height_val)}', (p.get_x() + p.get_width() / 2., height_val),
                            ha='center', va='bottom')
            file_path = os.path.join(self.output_dir, f"categorical_distribution_{column}.jpg")
            plt.savefig(file_path, bbox_inches='tight')
            plt.show()
            plt.close()

            column_summary = non_na_data.value_counts().reset_index()
            column_summary.columns = ['Элемент', 'Количество']
            column_summary.insert(0, 'Признак', column)
            category_summary.append(column_summary)

        if category_summary:
            formatted_summary = pd.concat(category_summary, ignore_index=True)
            formatted_summary['Признак'] = formatted_summary['Признак'].where(~formatted_summary['Признак'].duplicated(keep='first'), '')
            return formatted_summary
        else:
            print("Нет подходящих категориальных переменных для анализа.")
            return pd.DataFrame()

    @timing
    def plot_categorical_vs_target(self, target_column, categorical_columns):
        """
        Построить графики сравнения категориальных переменных с целевой переменной и сохранить их.
        Возвращает сводную таблицу распределения.
        """
        summary_data = []

        for column in categorical_columns:
            non_na_data = self.df[[column, target_column]].dropna()
            if non_na_data.empty:
                print(f"Пропущен график для столбца {column}, так как он пустой после удаления пропусков.")
                continue
            try:
                parsed_dates = pd.to_datetime(non_na_data[column], format='%d.%m.%Y', errors='coerce')
                if parsed_dates.notna().mean() > 0.8:
                    print(f"Пропущен график для столбца {column}, так как он содержит данные формата даты.")
                    continue
            except Exception as e:
                print(f"Ошибка при проверке формата даты для столбца {column}: {e}")
                continue
            if pd.api.types.is_numeric_dtype(non_na_data[column]):
                print(f"Пропущен график для столбца {column}, так как он содержит числовые данные.")
                continue

            unique_count = non_na_data[column].nunique()
            width = min(max(8, unique_count * 0.5), 20)
            rotation = 0 if unique_count <= 5 else 30 if unique_count <= 10 else 60
            height = 6

            plt.figure(figsize=(width, height))
            ax = sns.countplot(
                x=non_na_data[column].astype(str),
                hue=non_na_data[target_column].astype(str),
                palette="viridis"
            )
            plt.title(f"Распределение {column} в зависимости от {target_column}")
            plt.xlabel(column)
            plt.ylabel("Количество")
            plt.xticks(rotation=rotation)
            plt.legend(title=target_column)
            for p in ax.patches:
                height_val = p.get_height()
                ax.annotate(f'{int(height_val)}', (p.get_x() + p.get_width() / 2., height_val),
                            ha='center', va='bottom')
            file_path = os.path.join(self.output_dir, f"categorical_vs_target_{column}.jpg")
            plt.savefig(file_path, bbox_inches='tight')
            plt.show()
            plt.close()

            category_distribution = non_na_data.groupby([column, target_column]).size().reset_index(name='Количество')
            category_distribution.insert(0, 'Признак', column)
            summary_data.append(category_distribution)

        if summary_data:
            summary_df = pd.concat(summary_data, ignore_index=True)
            return summary_df
        else:
            print("Нет подходящих категориальных переменных для анализа.")
            return pd.DataFrame()

    @timing
    def plot_numeric_pairplot(self, numeric_columns=None):
        """
        Построить графики парных зависимостей для числовых признаков и сохранить в файл.
        """
        if numeric_columns is None:
            numeric_columns = self.df.select_dtypes(include=['number']).columns.tolist()

        if not numeric_columns:
            print("Нет числовых колонок для построения графиков парных зависимостей.")
            return

        pairplot_fig = sns.pairplot(self.df[numeric_columns], diag_kind="kde", corner=True)
        pairplot_fig.fig.suptitle("Графики парных зависимостей числовых признаков", y=1.02, fontsize=16)
        file_path = os.path.join(self.output_dir, "numeric_pairplot.jpg")
        pairplot_fig.savefig(file_path, bbox_inches='tight')
        plt.show()
        print(f"Графики парных зависимостей сохранены в: {file_path}")

    @timing
    def plot_kde_distributions(self, numeric_columns):
        """
        Построить гистограммы и графики KDE для числовых переменных с дополнительной аналитикой.
        Возвращает итоговую сводную таблицу с аналитикой по признакам.
        """
        summary_data = []

        for column in numeric_columns:
            data = self.df[column].dropna()
            if data.nunique() <= 1:
                print(f"Пропущен график для столбца {column}, так как все значения одинаковы или отсутствует дисперсия.")
                continue

            plt.figure(figsize=(10, 6))
            ax = sns.histplot(
                data,
                kde=True,
                bins=30,
                edgecolor="black",
                color="royalblue",
                alpha=0.8,
                linewidth=1.2
            )
            kde_color = "darkorange"
            sns.kdeplot(data, color=kde_color, linewidth=2, label="KDE (плотность)")
            plt.title(f"Гистограмма и KDE для признака: {column}", fontsize=14, fontweight="bold")
            plt.xlabel(column, fontsize=12)
            plt.ylabel("Плотность / Частота", fontsize=12)
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            median = data.median()
            mean = data.mean()
            std_dev = data.std()
            plt.axvline(median, color="red", linestyle="--", linewidth=2, label=f"Медиана: {median:.2f}")
            plt.axvline(mean, color="blue", linestyle="-.", linewidth=2, label=f"Среднее: {mean:.2f}")

            percentiles = [0.25, 0.75, 0.99]
            percentile_values = data.quantile(percentiles)
            perc_colors = ["green", "purple", "brown"]
            for perc, value, color in zip(percentiles, percentile_values, perc_colors):
                plt.axvline(value, color=color, linestyle=":", linewidth=2, label=f"{int(perc * 100)}-й перцентиль: {value:.2f}")

            try:
                x_range = np.linspace(data.min(), data.max(), 1000)
                ideal_pdf = norm.pdf(x_range, loc=mean, scale=std_dev)
                ideal_pdf_scaled = ideal_pdf * len(data) * (data.max() - data.min()) / 30
                plt.plot(x_range, ideal_pdf_scaled, color="orange", linestyle="--", linewidth=2.5, label="Идеальное распределение")
            except Exception as e:
                print(f"Ошибка при построении идеального распределения для столбца {column}: {e}")

            try:
                kde = gaussian_kde(data)
                kde_values = kde(x_range)
                peak_x = x_range[np.argmax(kde_values)]
                peak_y = kde_values.max()
                plt.annotate(f"Пик: {peak_x:.2f}", xy=(peak_x, peak_y),
                             xytext=(peak_x + 0.5, peak_y + 0.1),
                             arrowprops=dict(facecolor='black', arrowstyle="->"),
                             fontsize=10)
            except Exception as e:
                print(f"Не удалось построить KDE для столбца {column}: {e}")
                peak_x = None

            plt.legend(fontsize=10)
            file_path = os.path.join(self.output_dir, f"kde_distribution_{column}.jpg")
            plt.savefig(file_path, bbox_inches='tight')
            plt.show()
            plt.close()

            kurt = kurtosis(data)
            skewness = skew(data)
            if -0.5 <= skewness <= 0.5:
                distribution = "Нормальное"
            elif skewness > 0.5:
                distribution = "Смещенное вправо"
            elif skewness < -0.5:
                distribution = "Смещенное влево"
            else:
                distribution = "Неопределено"
            lower, upper = data.quantile(0.25), data.quantile(0.75)
            range_info = f"[{lower:.2f}, {upper:.2f}]"

            summary_data.append({
                "Признак": column,
                "Эксцесс": round(kurt, 2),
                "Асимметрия": round(skewness, 2),
                "Пик": round(peak_x, 2) if peak_x is not None else None,
                "Среднее": round(mean, 2),
                "Медиана": round(median, 2),
                "25-й перцентиль": round(percentile_values[0.25], 2),
                "75-й перцентиль": round(percentile_values[0.75], 2),
                "99-й перцентиль": round(percentile_values[0.99], 2),
                "Распределение": range_info,
                "Вывод": distribution
            })

        summary_df = pd.DataFrame(summary_data)
        return summary_df