import os
import pandas as pd
from utils import timing

class EDAExploratory:
    def __init__(self, df, output_dir="DATA_OUT/graphics"):
        """
        Инициализация класса для выполнения разведочного анализа данных.

        Параметры:
            df (pd.DataFrame): Входной DataFrame для анализа.
            output_dir (str): Директория для сохранения графиков.
        """
        self.df = df
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs("DATA_OUT", exist_ok=True)

    @timing
    def generate_eda_summary(self):
        """
        Сформировать сводную таблицу для разведочного анализа данных.

        Возвращает:
            dict: Словарь с ключами:
                - 'summary': DataFrame с информацией по столбцам.
                - 'duplicate_count': Общее количество дубликатов.
                - 'duplicates': DataFrame с дублированными строками (если есть).
                - 'duplicate_columns': Список пар дублированных столбцов.
        """
        summary = pd.DataFrame({
            'Название столбца': self.df.columns,
            'Пропущено строк': self.df.isnull().sum(),
            'Процент пропусков, %': (self.df.isnull().sum() / len(self.df) * 100).round(2),
            'Тип данных': self.df.dtypes,
            'Количество уникальных значений': [self.df[col].nunique() for col in self.df.columns],
            'Уникальные (категориальные) значения': [
                self.df[col].dropna().unique().tolist() if self.df[col].dtype in ['object', 'category', 'string'] else None
                for col in self.df.columns
            ],
            'Рекомендации': [
                "Удалить (константный столбец)" if self.df[col].nunique() == 1 else
                "Удалить (ID или уникальные значения)" if self.df[col].nunique() == len(self.df) else
                "Оставить"
                for col in self.df.columns
            ]
        }).reset_index(drop=True)

        duplicate_count = self.df.duplicated().sum()
        duplicates = self.df[self.df.duplicated()] if duplicate_count > 0 else pd.DataFrame()

        duplicate_columns = []
        for i, col1 in enumerate(self.df.columns):
            for col2 in self.df.columns[i + 1:]:
                if self.df[col1].equals(self.df[col2]):
                    duplicate_columns.append((col1, col2))

        return {
            'summary': summary,
            'duplicate_count': duplicate_count,
            'duplicates': duplicates,
            'duplicate_columns': duplicate_columns
        }

    @timing
    def find_rare_categories(self, categorical_columns, threshold=0.05):
        """
        Выявить редкие категории в категориальных переменных.

        Параметры:
            categorical_columns (list): Список категориальных колонок.
            threshold (float): Порог для определения редких категорий.
        Возвращает:
            dict: Словарь с редкими категориями для каждой переменной.
        """
        rare_categories = {}

        for col in categorical_columns:
            non_na_data = self.df[col].dropna()
            if non_na_data.empty:
                print(f"Пропущен анализ для столбца {col}, так как он пустой после удаления пропусков.")
                continue
            if pd.api.types.is_numeric_dtype(non_na_data):
                print(f"Пропущен анализ для столбца {col}, так как он содержит числовые данные.")
                continue
            try:
                parsed_dates = pd.to_datetime(non_na_data, format='%d.%m.%Y', errors='coerce')
                if parsed_dates.notna().mean() > 0.8:
                    print(f"Пропущен анализ для столбца {col}, так как он содержит данные формата даты.")
                    continue
            except Exception as e:
                print(f"Ошибка при проверке формата даты для столбца {col}: {e}")
                continue

            value_counts = non_na_data.value_counts(normalize=True)
            rare_values = value_counts[value_counts < threshold].index.tolist()
            rare_categories[col] = rare_values

        return rare_categories