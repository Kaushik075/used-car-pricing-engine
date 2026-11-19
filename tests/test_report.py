"""README results and the model card are generated from the saved metadata (runs everywhere, incl. CI)."""
import json

from src.config import METADATA_PATH
from src.report import model_card_markdown, pct, results_markdown

META = json.loads(METADATA_PATH.read_text())


def test_model_card_quotes_the_saved_results():
    card = model_card_markdown(META)
    assert pct(META["test_metrics"]["mape_pct"]) in card
    assert f"{META['test_range_coverage'][0]['by_estimate']:.1%}" in card
    assert f"{META['data']['clean_rows']:,}" in card
    for band in META["band_labels"]:
        assert band in card


def test_model_card_has_the_standard_sections():
    card = model_card_markdown(META)
    for heading in ("## Model details", "## Intended use", "## Training data", "## Performance (test set)",
                    "## Limitations", "## Monitoring and retraining", "## Reproduce"):
        assert heading in card


def test_readme_results_link_to_the_model_card():
    assert "[MODEL_CARD.md](MODEL_CARD.md)" in results_markdown(META)
