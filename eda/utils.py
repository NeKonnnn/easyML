import time
import os
import pandas as pd

def timing(func):
    """
    Декоратор для измерения времени выполнения функций.
    """
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Функция '{func.__name__}' выполнена за {elapsed_time:.2f} секунд.")
        return result
    return wrapper

def save_all_summaries_to_excel(summaries, file_path="DATA_OUT/eda_information.xlsx"):
    """
    Сохранить все сводные таблицы в один Excel файл,
    где каждая таблица находится на отдельном листе.

    Параметры:
        summaries (dict): Словарь с именами листов в качестве ключей и DataFrame в качестве значений.
        file_path (str): Путь к файлу Excel.
    """
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with pd.ExcelWriter(file_path, engine="xlsxwriter") as writer:
        for sheet_name, df in summaries.items():
            valid_sheet_name = sheet_name[:31]  # Ограничение длины имени листа
            df.to_excel(writer, sheet_name=valid_sheet_name, index=False)
    print(f"Все сводные таблицы сохранены в файл: {file_path}")