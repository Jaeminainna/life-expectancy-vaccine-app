# 💉 VaccineLife - Life Expectancy Prediction Dashboard

A Streamlit-based data product that predicts life expectancy based on vaccine coverage rates and provides comprehensive data visualizations.

## 🌟 Features

### Multiple User Flows
The application offers three distinct flows to accommodate different user needs:

1. **📊 Visualization Only** - Upload your data and explore various visualizations without needing a trained model
2. **🔮 Prediction Only** - Load your pre-trained model and make predictions on vaccine coverage scenarios
3. **📊🔮 Both** - Full functionality with both visualization and prediction capabilities

### Visualizations Included
- **Global Overview**: Life expectancy distribution and trends over time
- **Vaccine Coverage Analysis**: Average coverage by vaccine type and trends
- **Correlation Analysis**: Vaccine vs life expectancy correlations and heatmaps
- **Country Analysis**: Country-specific trends and top/bottom rankings
- **Scatter Plot Analysis**: Interactive scatter plots with trendlines

### Prediction Features
- Input vaccine coverage values for 16 different vaccines
- Quick-fill options (Low/Medium/High coverage)
- Visual prediction results with interpretation
- Supports pre-trained models saved with joblib

## 🚀 Installation

1. Clone or download this project

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the application:
```bash
streamlit run app.py
```

## 📁 File Structure

```
vaccine_life_expectancy_app/
├── app.py                 # Main Streamlit application
├── requirements.txt       # Python dependencies
├── README.md             # This file
└── models/               # (Optional) Store your trained models here
    ├── model.joblib
    ├── scaler.joblib
    ├── imputer.joblib
    └── feature_names.joblib
```

## 🤖 Model Requirements

### Saving Your Model from Google Colab

When training your model in Google Colab, save the following artifacts using joblib:

```python
import joblib

# After training your model
joblib.dump(model, 'model.joblib')
joblib.dump(scaler, 'scaler.joblib')  # Optional: if you used StandardScaler
joblib.dump(imputer, 'imputer.joblib')  # Optional: if you used SimpleImputer
joblib.dump(feature_names, 'feature_names.joblib')  # List of feature column names
```

### Expected Feature Names
The app expects these vaccine columns (or a subset based on your model):
- BCG, DTP1, DTP3, HEPB3, HEPBB, HIB3
- IPV1, IPV2, MCV1, MCV2, MENGA, PCV3
- POL3, RCV1, ROTAC, YFV

## 📊 Data Format

### Input Dataset
Your CSV/TXT/Excel file should contain:
- `country` - Country name
- `country_code` - ISO country code (optional)
- `year` - Year of observation
- Vaccine columns (BCG, DTP1, DTP3, etc.) - Coverage percentages (0-100)
- `life_expectancy` - Life expectancy in years

Example:
```csv
country,year,BCG,DTP1,DTP3,MCV1,life_expectancy
Afghanistan,2020,72.0,70.0,61.0,57.0,61.454
Algeria,2020,99.0,94.0,84.0,80.0,73.257
```

## 🎯 How to Use

### For Visualization Only:
1. Select "📊 Visualization Only" in the sidebar
2. Upload your dataset (CSV/TXT/Excel)
3. Navigate through the tabs to explore different visualizations

### For Prediction Only:
1. Select "🔮 Prediction Only" in the sidebar
2. Upload your trained model (.joblib or .pkl)
3. Optionally upload scaler, imputer, and feature names
4. Click "Load Model"
5. Enter vaccine coverage values and click "Predict"

### For Both:
1. Select "📊🔮 Both" in the sidebar
2. Upload your dataset for visualizations
3. Upload your trained model for predictions
4. Navigate between tabs for different features

## 🛠️ Customization

### Adding New Vaccines
Edit the `VACCINE_INFO` dictionary in `app.py` to add new vaccine types:

```python
VACCINE_INFO = {
    'NEW_VACCINE': 'Description of the new vaccine',
    # ... existing vaccines
}
```

### Changing Theme
The app uses custom CSS for styling. Modify the `st.markdown()` section with the `<style>` tags to customize colors and appearance.

## 📝 Notes

- The app uses Plotly for interactive visualizations
- All charts have dark theme styling
- Session state is used to persist data and model between interactions
- Temporary files are cleaned up after model loading

## 🐛 Troubleshooting

**Model not loading?**
- Ensure your model was saved with `joblib.dump()`
- Check that scikit-learn versions match between training and deployment

**Visualizations not showing?**
- Verify your dataset has the expected column names
- Check that numeric columns contain valid numbers

**Prediction errors?**
- Ensure feature names match between model training and input
- Upload the feature_names.joblib file if available

## 📄 License

MIT License - Feel free to use and modify for your projects.
