import numpy as np
import pandas as pd
import pytest

from src.data import swat

SENSORS = ["FIT101", "LIT101", "MV101", "P101", "AIT201", "DPIT301", "PIT501", "UV401"]


def _frame(start, n, attack_rows=(), seed=0):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(rng.normal(size=(n, len(SENSORS))), columns=SENSORS)
    times = pd.date_range(start, periods=n, freq="s")
    df.insert(0, " Timestamp", [f" {t.day}/{t.month}/{t.year} {t:%I:%M:%S %p}" for t in times])
    label = np.array(["Normal"] * n, dtype=object)
    label[list(attack_rows)] = "A ttack"  # spelling found in the official attack file
    df["Normal/Attack"] = label
    return df


def test_sensor_type_and_stage():
    assert swat.get_sensor_type("DPIT301") == "Diff_Pressure"
    assert swat.get_sensor_type("PIT501") == "Pressure"
    assert swat.get_sensor_type("P602") == "Pump"
    assert swat.get_sensor_type("UV401") == "UV_Dechlorinator"
    assert swat.get_stage("FIT101") == 1 and swat.get_stage("P602") == 6


def test_kaggle_merged_rebuilds_official_layout(tmp_path):
    drain = swat.DRAIN_SECONDS
    v0 = _frame("2015-12-28 08:00:00", drain + 300)            # normal week incl. drain
    v1 = v0.iloc[drain:].reset_index(drop=True)                 # same week without drain
    attack = _frame(swat.A1_ATTACK_PERIOD_START, 200, attack_rows=range(50, 80), seed=1)
    v0 = v0.copy()
    v0[["MV101", "AIT201"]] = np.nan                            # as in the Kaggle v0 block
    merged = pd.concat([attack, v0, v1], ignore_index=True)     # out of order, v0 first
    path = tmp_path / "merged.csv"
    merged.to_csv(path, index=False)

    df = swat.load_swat_kaggle_merged(str(path))
    assert len(df) == 300 + 200
    assert df["DATETIME"].is_monotonic_increasing and not df["DATETIME"].duplicated().any()
    assert (df["PHASE"] == "normal_week").sum() == 300
    assert df.loc[df["PHASE"] == "attack_period", "ATT_FLAG"].sum() == 30
    assert list(df.columns) == ["DATETIME"] + SENSORS + ["ATT_FLAG", "PHASE"]
    assert not df[SENSORS].isna().any().any()                   # the complete v1 rows won
    assert df.attrs["dropped_columns"] == []


def test_kaggle_drops_components_missing_from_whole_normal_week(tmp_path):
    normal = _frame("2015-12-28 08:00:00", swat.DRAIN_SECONDS + 100)
    normal["MV101"] = np.nan
    attack = _frame(swat.A1_ATTACK_PERIOD_START, 50, attack_rows=[3])
    pd.concat([normal, attack], ignore_index=True).to_csv(tmp_path / "m.csv", index=False)
    df = swat.load_swat_kaggle_merged(str(tmp_path / "m.csv"))
    assert df.attrs["dropped_columns"] == ["MV101"] and "MV101" not in df.columns
    with pytest.raises(ValueError, match="missing sensor values"):
        swat.load_swat_kaggle_merged(str(tmp_path / "m.csv"), drop_columns_missing_in_normal=False)


def test_official_loader_drops_drain(tmp_path):
    v0 = _frame("2015-12-28 08:00:00", swat.DRAIN_SECONDS + 100)
    attack = _frame(swat.A1_ATTACK_PERIOD_START, 50, attack_rows=[3])
    v0.to_csv(tmp_path / "n.csv", index=False)
    attack.to_csv(tmp_path / "a.csv", index=False)
    df = swat.load_swat_official(str(tmp_path / "n.csv"), str(tmp_path / "a.csv"))
    assert len(df) == 150 and df["ATT_FLAG"].sum() == 1


def test_windows_match_loop_reference_and_fit_on_train_only():
    df = swat._finalize(swat._standardize(_frame("2015-12-28 10:00:00", 400, attack_rows=range(300, 320)),
                                          "attack_period"))
    fit = np.arange(200)
    graphs, labels, extra = swat.build_windowed_graphs(df, window_size=20, stride=10, fit_rows=fit,
                                                       topology="process")
    sensors = extra["sensor_list"]
    s = extra["starts"][7]
    window = df[sensors].iloc[s:s + 20].astype(float)
    fit_df = df.iloc[fit]
    vmin, vmax = fit_df[sensors].min(), fit_df[sensors].max()
    expected_mean = (window.mean() - vmin) / (vmax - vmin + 1e-9)
    expected_std = window.std() / (vmax - vmin + 1e-9)
    np.testing.assert_allclose(graphs[7].x[:, 0].numpy(), expected_mean.to_numpy(), rtol=1e-4, atol=1e-5)
    np.testing.assert_allclose(graphs[7].x[:, 1].numpy(), expected_std.to_numpy(), rtol=1e-4, atol=1e-5)
    assert labels.tolist() == [int(df["ATT_FLAG"].iloc[s:s + 20].any()) for s in extra["starts"]]
    assert graphs[0].edge_index.shape[1] == 2 * extra["n_edges"]


def test_standard_protocol_rows():
    df = pd.DataFrame({"PHASE": ["normal_week"] * 100 + ["attack_period"] * 40})
    rows = swat.standard_protocol_rows(df, val_frac=0.1)
    assert len(rows["fit"]) == 90 and len(rows["val"]) == 10 and len(rows["test"]) == 40
    assert rows["fit"].max() < rows["val"].min() < rows["test"].min()


def test_unknown_label_rejected(tmp_path):
    bad = _frame("2015-12-28 10:00:00", 10)
    bad.loc[3, "Normal/Attack"] = "Maybe"
    with pytest.raises(ValueError):
        swat._standardize(bad, "attack_period")
