# Car Rental Prediction System

A Flask-based web application that uses machine learning to predict daily rental rates for vehicles. Features a simplified architecture with CSV data source, SQLite prediction history, and Bootstrap UI.

## 🚀 Features

- **Machine Learning Predictions**: Predict car rental prices using a trained ML model
- **CSV Data Source**: All vehicle and location data loaded from CarRentalData.csv (5,852 records)
- **Prediction History**: SQLite database stores only prediction history for tracking
- **Modern UI**: Bootstrap 5 with responsive design and user-friendly interface
- **Error Handling**: Comprehensive error pages with troubleshooting guidance
- **Data Visualization**: Charts comparing predicted prices with similar vehicles
- **Data Overview**: Interactive modal showing dataset statistics

## 📋 System Architecture

- **Data Source**: `CarRentalData.csv` (5,852 rental records)
- **Database**: SQLite with single `PredictionHistory` table
- **ML Model**: Pre-trained model in `car_rental_model.pkl` (37MB)
- **Frontend**: Flask templates with Bootstrap 5 UI

## 🛠️ Installation

1. **Navigate to the project directory**:
   ```bash
   cd car_rental_system
   ```

2. **Create a virtual environment** (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## 🏃 Running the Application

1. **Start the Flask application**:
   ```bash
   python app.py
   ```

2. **Open your browser** and navigate to:
   ```
   http://localhost:5000
   ```

The database will be automatically created with the prediction history table on first run.

## 🎯 Usage

### Making Predictions

1. **Dashboard**: Navigate to the main page to access the prediction form
2. **Fill Vehicle Details**: Select make, model, year, type, and fuel type from CSV data
3. **Choose Location**: Pick city, state, country, and coordinates from available data
4. **Set Parameters**: Enter rating, trips taken, and review count
5. **Predict**: Click "Predict Price" to get estimated daily rental rate
6. **View Results**: See prediction with comparison chart and detailed information

### Viewing Data

- **Data Overview**: Click "Data Overview" in navigation for dataset statistics
- **Prediction History**: View all past predictions with input details
- **Error Handling**: User-friendly error pages when predictions fail

## 📊 Dataset Information

The system uses `CarRentalData.csv` containing:
- **5,852 rental records**
- **1,500+ unique vehicles** across multiple makes/models
- **1,000+ locations** across various cities and states
- **Price range**: $28 - $600 daily rates
- **Vehicle years**: 2005-2020
- **Fuel types**: Gasoline, Electric, Hybrid, Diesel

## 🏗️ Technical Details

### Database Schema (Simplified)

- **PredictionHistory**: 
  - `id` (Primary Key)
  - `predicted_price` (Float)
  - `vehicle_info` (String)
  - `prediction_date` (DateTime)
  - `input_data` (JSON Text)

### Machine Learning

- **Model**: Pre-trained RandomForest from `car_rental_model.pkl`
- **Features**: Defined in `car_rental_model_features.json`
- **Encoding**: Automatic categorical variable encoding
- **Fallback**: Creates backup model if main model fails
- **Validation**: Input validation and error handling

### File Structure

```
car_rental_system/
├── app.py                          # Main Flask application
├── requirements.txt                # Python dependencies  
├── README.md                      # Documentation
├── car_rental_model.pkl           # Pre-trained ML model (37MB)
├── car_rental_model_features.json # Model features
├── CarRentalData.csv             # Complete dataset (5,852 records)
├── instance/
│   └── car_rental.db              # SQLite database (auto-created)
└── templates/                     # HTML templates
    ├── base.html                  # Base template with navigation
    ├── index.html                 # Dashboard with prediction form
    ├── history.html               # Prediction history view
    ├── prediction_result.html     # Detailed prediction results
    └── prediction_error.html      # Error handling page
```

## 🔧 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | Dashboard with prediction form |
| `POST` | `/predict` | Make price prediction |
| `GET` | `/prediction/<id>` | View specific prediction |
| `GET` | `/history` | Prediction history |
| `GET` | `/data_overview` | Dataset statistics (JSON) |

## 🎨 UI Features

- **Responsive Design**: Works on desktop, tablet, and mobile
- **Interactive Forms**: Dropdowns populated from CSV data
- **Error Handling**: User-friendly error pages with guidance
- **Data Visualization**: Price comparison charts
- **Statistics Cards**: Real-time data overview
- **Navigation**: Clean, modern interface

## ⚙️ Configuration

### Database Configuration
```python
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///car_rental.db'
```

### CSV Data Source
The system automatically loads data from `CarRentalData.csv`. Ensure this file is present in the project root.

## 🚨 Error Handling

The system includes comprehensive error handling:

- **Input Validation**: Checks for valid data types and ranges
- **Model Errors**: Graceful handling when ML model fails
- **Database Issues**: Continues predictions even if history save fails
- **Missing Data**: Clear messages for incomplete information
- **User Guidance**: Step-by-step troubleshooting tips

## 🔍 Troubleshooting

### Common Issues

1. **CSV File Missing**
   ```bash
   Error: Dataset file not found at CarRentalData.csv
   ```
   **Solution**: Ensure `CarRentalData.csv` is in the project root directory.

2. **Model Loading Error**
   ```bash
   Error loading model: [error details]
   ```
   **Solution**: The system will create a fallback model automatically using CSV data.

3. **Database Issues**
   ```bash
   sqlite3.OperationalError: [database error]
   ```
   **Solution**: Delete `instance/car_rental.db` and restart the application.

4. **Missing Dependencies**
   ```bash
   ModuleNotFoundError: No module named 'flask'
   ```
   **Solution**: Install requirements: `pip install -r requirements.txt`

5. **Port Already in Use**
   ```bash
   OSError: [Errno 48] Address already in use
   ```
   **Solution**: Kill existing Flask process or use different port.

## 🛠️ Development

### Adding Features

1. **Database Models**: Modify `PredictionHistory` model in `app.py`
2. **Templates**: Edit HTML files in `templates/` directory
3. **Styling**: Update CSS in `base.html` template
4. **ML Features**: Modify `car_rental_model_features.json`

### Testing

The system includes built-in validation:
- Input data validation
- Model prediction verification
- Error boundary testing
- CSV data integrity checks

## 📈 Performance

- **CSV Loading**: ~1 second for 5,852 records
- **Prediction Time**: ~100ms per prediction
- **Database Operations**: Minimal overhead (history only)
- **Memory Usage**: ~50MB including ML model

## 📝 License

This project is for educational and demonstration purposes. The machine learning model and sample data should be used responsibly and in accordance with data usage policies.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📞 Support

For issues and questions:
1. Check the troubleshooting section
2. Review error messages in the UI
3. Check application logs for technical details
4. Ensure all dependencies are correctly installed 