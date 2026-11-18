# Data

`Car_Price_data.csv` is **not included in this repository**.

It is course-provided data (upGrad / IIIT Bangalore, AutoScout24 used-car listings,
15,915 rows x 23 columns, target `price`). It came without a licence that allows
redistribution, so it is kept out of the public repo.

To reproduce the training:

1. Put your copy of `Car_Price_data.csv` in this folder (`data/Car_Price_data.csv`).
2. From the project root run `python -m src.train`.

The trained model in `models/` is included, so the Streamlit app runs without the CSV.
It stores fitted parameters and aggregated defaults (medians, most common values), not raw listings.
