import pandas as pd
import pytest

import app


def raw_dataframe():
    return pd.DataFrame(
        {
            app.USAGE: [1.0, 3.0, 5.0, 7.0, 2.0, 4.0, 6.0, 8.0],
            app.PRODUCTIVITY: [9, 8, 7, 6, 9, 8, 7, 5],
            app.NIGHT: [
                "Never",
                "Rarely",
                "Sometimes",
                "Often",
                "Daily",
                "Never",
                "Sometimes",
                "Daily",
            ],
            app.SLEEP: [
                "Very good",
                "Good",
                "Fair",
                "Poor",
                "Very poor",
                "Good",
                "Fair",
                "Poor",
            ],
            app.FOCUS: [
                "Never",
                "Rarely",
                "Sometimes",
                "Yes",
                "Yes",
                "Rarely",
                "Sometimes",
                "Yes",
            ],
            app.DOOM: [
                "Never",
                "Rarely",
                "Sometimes",
                "Often",
                "Daily",
                "Never",
                "Often",
                "Daily",
            ],
            app.FOMO: [1, 3, 5, 8, 2, 6, 7, 10],
            app.AGE: [20, 28, 36, 48, 60, 23, 33, 43],
        }
    )


def test_prepare_dataframe_maps_values_and_creates_categories():
    prepared = app.prepare_dataframe(raw_dataframe())

    assert prepared is not None
    assert len(prepared) == 8
    assert prepared[app.SLEEP].tolist() == [5, 4, 3, 2, 1, 4, 3, 2]
    assert prepared[app.NIGHT].tolist() == [0, 1, 2, 3, 4, 0, 2, 4]
    assert prepared[app.FOCUS].tolist() == [0, 1, 2, 3, 3, 1, 2, 3]
    assert prepared["Uso noturno"].cat.categories.tolist() == app.NIGHT_ORDER
    assert prepared["Intensidade de uso"].cat.categories.tolist() == app.USAGE_ORDER
    assert prepared["Nível de FOMO"].cat.categories.tolist() == app.FOMO_ORDER
    assert prepared["Faixa etária"].cat.categories.tolist() == app.AGE_ORDER


def test_prepare_dataframe_discards_rows_with_invalid_required_values():
    data = raw_dataframe()
    data.loc[0, app.SLEEP] = "Unknown"
    data[app.PRODUCTIVITY] = data[app.PRODUCTIVITY].astype(object)
    data.loc[1, app.PRODUCTIVITY] = "not a number"

    prepared = app.prepare_dataframe(data)

    assert prepared is not None
    assert len(prepared) == 6


def test_prepare_dataframe_returns_none_for_empty_input():
    assert app.prepare_dataframe(pd.DataFrame()) is None
    assert app.prepare_dataframe(None) is None


def test_prepare_dataframe_reports_missing_columns(monkeypatch):
    errors = []
    monkeypatch.setattr(app.st, "error", errors.append)

    prepared = app.prepare_dataframe(raw_dataframe().drop(columns=[app.FOMO]))

    assert prepared is None
    assert errors == ["Colunas ausentes: Pontuação de FOMO"]


def test_build_charts_returns_expected_figures_and_correlation():
    prepared = app.prepare_dataframe(raw_dataframe())

    scatter, box, focus_bar, heatmap, correlation = app.build_charts(prepared)

    assert [figure.data[0].type for figure in (scatter, box, focus_bar, heatmap)] == [
        "scatter",
        "box",
        "bar",
        "heatmap",
    ]
    assert scatter.layout.height == 340
    assert correlation < 0


def test_optional_chart_builders_handle_missing_or_empty_age_data():
    prepared = app.prepare_dataframe(raw_dataframe())
    without_age = prepared.drop(columns=[app.AGE, "Faixa etária"])

    assert app.build_age_charts(without_age) == (None, None)
    assert app.build_age_charts(prepared[prepared["Faixa etária"].isna()]) == (None, None)


def test_dependency_charts_return_none_for_empty_fomo_data():
    prepared = app.prepare_dataframe(raw_dataframe())
    without_fomo = prepared.iloc[0:0].copy()

    assert app.build_dependency_charts(without_fomo) == (None, None)
