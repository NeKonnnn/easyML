import os
import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency, f_oneway
from utils import timing

class EDACorrelations:
    def __init__(self, df, output_dir="DATA_OUT/graphics"):
        """
        Инициализация класса для анализа корреляций.

        Параметры:
            df (pd.DataFrame): Входной DataFrame.
            output_dir (str): Директория для сохранения графиков (если требуется).
        """
        self.df = df
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs("DATA_OUT", exist_ok=True)

    @timing
    def analyze_categorical_cross_tabulations(self, categorical_columns):
        """
        Построить объединённую таблицу сопряженности для пар категориальных переменных
        и сформировать автоматические выводы о взаимосвязях.
        Возвращает объединённый DataFrame и словарь с выводами.
        """
        combined_cross_tab = []
        conclusions = {}

        for i, col1 in enumerate(categorical_columns):
            col1_data = self.df[col1].dropna()
            try:
                parsed_dates = pd.to_datetime(col1_data, format='%d.%m.%Y', errors='coerce')
                if parsed_dates.notna().mean() > 0.8:
                    print(f"Пропущен анализ для столбца {col1}, так как он содержит данные формата даты.")
                    continue
            except Exception as e:
                print(f"Ошибка при проверке формата даты для столбца {col1}: {e}")
                continue

            for col2 in categorical_columns[i + 1:]:
                col2_data = self.df[col2].dropna()
                try:
                    parsed_dates = pd.to_datetime(col2_data, format='%d.%m.%Y', errors='coerce')
                    if parsed_dates.notna().mean() > 0.8:
                        print(f"Пропущен анализ для столбца {col2}, так как он содержит данные формата даты.")
                        continue
                except Exception as e:
                    print(f"Ошибка при проверке формата даты для столбца {col2}: {e}")
                    continue

                ctab = pd.crosstab(self.df[col1], self.df[col2], dropna=True)
                ctab.index = pd.MultiIndex.from_product([[col1], ctab.index], names=["Признак 1", "Категории признака 1"])
                ctab.columns = pd.MultiIndex.from_product([[col2], ctab.columns], names=["Признак 2", "Категории признака 2"])
                combined_cross_tab.append(ctab)
                unique_combinations = (ctab.sum(axis=1) == 1).all() and (ctab.sum(axis=0) == 1).all()
                if unique_combinations:
                    conclusions[f"{col1} vs {col2}"] = "Чёткая взаимосвязь 1:1 (каждой категории одной переменной соответствует только одна категория другой переменной)."
                else:
                    conclusions[f"{col1} vs {col2}"] = "Существуют неоднозначные связи (одна категория соответствует нескольким категориям другой переменной)."

        if combined_cross_tab:
            combined_cross_tab_df = pd.concat(combined_cross_tab, axis=0)
            return combined_cross_tab_df, conclusions
        else:
            print("Нет подходящих категориальных переменных для анализа.")
            return pd.DataFrame(), conclusions

    @timing
    def analyze_correlations(self, target_column, threshold=0.5, correlation_types=None, include_phi=False, include_cramers_v=False):
        """
        Анализ корреляций между признаками и целевой переменной.
        Возвращает словарь с результатами для числовых признаков (корреляции),
        результатов теста ANOVA (для категориального таргета) и Cramér's V.
        """
        results = {}
        if correlation_types is None:
            correlation_types = ['pearson', 'spearman', 'kendall']

        if target_column is None or target_column not in self.df.columns:
            raise ValueError("Укажите корректное название целевой переменной.")

        self.df = self.df.dropna()
        numeric_columns = self.df.select_dtypes(include=['number']).columns.tolist()
        categorical_columns = self.df.select_dtypes(include=['object', 'category']).columns.tolist()
        if target_column not in numeric_columns + categorical_columns:
            raise ValueError("Целевая переменная должна быть числовой или категориальной.")

        def cheddock_scale(value):
            if value < 0.1:
                return "Очень слабая связь"
            elif 0.1 <= value < 0.3:
                return "Слабая связь"
            elif 0.3 <= value < 0.5:
                return "Умеренная связь"
            elif 0.5 <= value < 0.7:
                return "Заметная связь"
            elif 0.7 <= value < 0.9:
                return "Высокая связь"
            elif 0.9 <= value < 1.0:
                return "Весьма высокая связь"
            else:
                return "Идеальная связь"

        def cramers_v_scale(value):
            if value <= 0.2:
                return "Слабая связь"
            elif 0.2 < value <= 0.6:
                return "Умеренная связь"
            elif value > 0.6:
                return "Сильная связь"
            else:
                return "Нет связи"

        if target_column in numeric_columns:
            correlations = {}
            if 'pearson' in correlation_types:
                correlations['pearson'] = self.df[numeric_columns].corr(method='pearson')
            if 'spearman' in correlation_types:
                correlations['spearman'] = self.df[numeric_columns].corr(method='spearman')
            if 'kendall' in correlation_types:
                correlations['kendall'] = self.df[numeric_columns].corr(method='kendall')

            correlated_pairs = []
            processed_pairs = set()

            for col1 in numeric_columns:
                for col2 in numeric_columns:
                    if col1 != col2 and (col1, col2) not in processed_pairs and (col2, col1) not in processed_pairs:
                        pair_data = {"Признак 1": col1, "Признак 2": col2}
                        for corr_type in correlation_types:
                            if corr_type in correlations:
                                corr_value = abs(correlations[corr_type].loc[col1, col2])
                                pair_data[f"Корреляция {corr_type.capitalize()}"] = round(corr_value, 2)
                                pair_data[f"Вывод по шкале Чеддока ({corr_type.capitalize()})"] = cheddock_scale(corr_value)
                        if include_phi or include_cramers_v:
                            contingency_table = pd.crosstab(self.df[col1].round(), self.df[col2].round())
                            chi2, _, _, _ = chi2_contingency(contingency_table)
                            n = contingency_table.sum().sum()
                            rows, cols = contingency_table.shape
                            if rows == 2 and cols == 2 and include_phi:
                                phi_coefficient = np.sqrt(chi2 / n) if n > 0 else 0
                                pair_data["Phi-коэффициент"] = round(phi_coefficient, 2)
                            if include_cramers_v:
                                min_dim = min(rows - 1, cols - 1)
                                cramers_v = np.sqrt(chi2 / (n * min_dim)) if min_dim > 0 else None
                                pair_data["Cramer's V"] = round(cramers_v, 2) if cramers_v is not None else None
                        if any(pair_data.get(f"Корреляция {corr_type.capitalize()}") >= threshold for corr_type in correlation_types):
                            correlated_pairs.append(pair_data)
                            processed_pairs.add((col1, col2))
            correlated_df = pd.DataFrame(correlated_pairs).drop_duplicates(subset=["Признак 1", "Признак 2"])
            if correlation_types:
                primary_corr_type = correlation_types[0]
                sort_column = f"Корреляция {primary_corr_type.capitalize()}"
                if sort_column in correlated_df.columns:
                    correlated_df = correlated_df.sort_values(by=sort_column, ascending=False)
            results['correlations'] = correlated_df

        elif target_column in categorical_columns:
            anova_results = []
            for col in numeric_columns:
                unique_groups = self.df[target_column].dropna().unique()
                if len(unique_groups) > 1:
                    groups = [self.df[col][self.df[target_column] == group] for group in unique_groups]
                    f_stat, p_value = f_oneway(*groups)
                    anova_results.append({
                        "Признак": col,
                        "F-статистика": round(f_stat, 2),
                        "P-значение": round(p_value, 4)
                    })
            results['anova'] = pd.DataFrame(anova_results)

            cramers_v_results = []
            for col in categorical_columns:
                if col != target_column:
                    contingency_table = pd.crosstab(self.df[col], self.df[target_column])
                    chi2, _, _, _ = chi2_contingency(contingency_table)
                    n = contingency_table.sum().sum()
                    rows, cols = contingency_table.shape
                    min_dim = min(rows - 1, cols - 1)
                    cramers_v = np.sqrt(chi2 / (n * min_dim)) if min_dim > 0 else None
                    cramers_v_results.append({
                        "Признак": col,
                        "Cramer's V": round(cramers_v, 2) if cramers_v is not None else None,
                        "Интерпретация (Cramer's V)": cramers_v_scale(cramers_v) if cramers_v is not None else "Нет связи"
                    })
            results['cramers_v'] = pd.DataFrame(cramers_v_results)

        return results

    @timing
    def analyze_phik_correlations(self, threshold=0.5):
        """
        Анализ Phik корреляций между всеми признаками.
        Возвращает DataFrame с парами признаков, их Phik-коэффициентом и выводами.
        """
        self.df = self.df.dropna()
        phik_corr = self.df.phik_matrix(interval_cols=None)

        def interpret_phik(value):
            if value <= 0.2:
                return "Слабая связь"
            elif 0.2 < value <= 0.4:
                return "Умеренная связь"
            elif 0.4 < value <= 0.6:
                return "Заметная связь"
            elif 0.6 < value <= 0.8:
                return "Высокая связь"
            else:
                return "Очень высокая связь"

        correlated_pairs = []
        processed_pairs = set()

        for col1 in phik_corr.columns:
            for col2 in phik_corr.index:
                if col1 != col2 and (col1, col2) not in processed_pairs and (col2, col1) not in processed_pairs:
                    phik_value = abs(phik_corr.loc[col1, col2])
                    if phik_value >= threshold:
                        correlated_pairs.append({
                            "Признак 1": col1,
                            "Признак 2": col2,
                            "Phik-коэффициент": round(phik_value, 2),
                            "Вывод": interpret_phik(phik_value)
                        })
                        processed_pairs.add((col1, col2))
        phik_correlations_df = pd.DataFrame(correlated_pairs).drop_duplicates(subset=["Признак 1", "Признак 2"])
        return phik_correlations_df