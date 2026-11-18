import numpy as np
import pandas as pd

from src.features import FeatureListBinarizer, KmImputer


def test_binarizer_learns_vocabulary_from_fit_data_only():
    train = pd.DataFrame({"Extras": ["Alloy wheels,Sport seats", "Alloy wheels", "Alloy wheels,Ski bag"]})
    # min_share=0.5 -> an item must be on at least half the training cars
    enc = FeatureListBinarizer(min_share=0.5).fit(train)
    assert enc.vocabulary_["Extras"] == ["Alloy wheels"]

    # "Roof rack" was never seen in training: ignored, no crash
    new = pd.DataFrame({"Extras": ["Roof rack,Alloy wheels", ""]})
    out = enc.transform(new)
    assert out.tolist() == [[1.0], [0.0]]
    assert list(enc.get_feature_names_out()) == ["Extras__Alloy wheels"]


def test_binarizer_handles_missing_values():
    train = pd.DataFrame({"Extras": ["A", None, "A,B"]})
    enc = FeatureListBinarizer(min_share=0.3).fit(train)
    assert enc.transform(pd.DataFrame({"Extras": [np.nan]})).tolist() == [[0.0, 0.0]]


def test_km_imputer_uses_type_median_from_training_rows():
    train = pd.DataFrame(
        {"km": [10.0, 20.0, 1647.36, 30000.0, 50000.0], "Type": ["New", "New", "New", "Used", "Used"]}
    )
    imp = KmImputer().fit(train)
    # the fractional value is ignored when learning: New median is 15, not 20
    assert imp.type_medians_["New"] == 15.0

    out = imp.transform(pd.DataFrame({"km": [1647.36, 12345.0], "Type": ["New", "Used"]}))
    assert out[0].tolist() == [15.0, 1.0]  # filled, flagged
    assert out[1].tolist() == [12345.0, 0.0]  # real value kept


def test_km_imputer_unknown_type_falls_back_to_global_median():
    train = pd.DataFrame({"km": [100.0, 300.0], "Type": ["New", "Used"]})
    imp = KmImputer().fit(train)
    out = imp.transform(pd.DataFrame({"km": [np.nan], "Type": ["Vintage"]}))
    assert out[0].tolist() == [200.0, 1.0]
