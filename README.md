# 🚗 AUTO.RIA Price Prediction

Predicting used car prices on the Ukrainian market using data scraped from auto.ria.com.

## Problem
Given a used car listing (brand, year, mileage, fuel type, transmission, accident history),
predict its price in USD.

## Dataset
- **Source:** scraped from auto.ria.com using requests + BeautifulSoup
- **Size:** ~3,900 listings after cleaning (removed "new car" dealer listings)
- **Features:** release year, mileage, brand, fuel type, engine volume, 
  transmission, accident history, region

## Methodology
1. **Scraping** (`notebooks/01_scraping.ipynb`) — custom parser, 50 pages, ~5k listings
2. **EDA & Preprocessing** (`notebooks/02_eda.ipynb`) — feature extraction, OHE, 
   log-transform of target variable
3. **Modeling** (`notebooks/03_modeling.ipynb`) — Linear → Ridge → Lasso → LightGBM

## Results

| Model | MAE | RMSE | R² |
|-------|-----|------|----|
| Linear Regression | $7,616 | $18,883 | 0.743 |
| Ridge | $7,620 | $18,904 | 0.742 |
| Lasso | $7,720 | $19,322 | 0.738 |
| **LightGBM** | **$4,694** | **$9,671** | **0.860** |

LightGBM outperforms linear models by 2x on MAE.  
Top features by SHAP: `release_year` > `engine_volume` > `mileage`.

## How to Run
```bash
git clone https://github.com/YOUR_USERNAME/auto-ria-price-prediction
cd auto-ria-price-prediction
poetry install
poetry run jupyter notebook
```

To launch Streamlit demo:
```bash
poetry run streamlit run app.py
```

## Stack
`Python` `pandas` `scikit-learn` `LightGBM` `SHAP` `BeautifulSoup` `Streamlit`
