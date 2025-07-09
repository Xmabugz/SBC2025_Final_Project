# -*- coding: utf-8 -*-
from flask import Flask, redirect, url_for, request, flash, render_template, jsonify
from flask_bootstrap import Bootstrap5
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from sqlalchemy import Integer, String, Text, Float, DateTime
from flask_hashing import Hashing
import pandas as pd
import joblib
import numpy as np
import json
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for web applications
import matplotlib.pyplot as plt
import io
import base64
from datetime import datetime
import os

app = Flask(__name__)
bootstrap = Bootstrap5(app)
hashing = Hashing(app)

# Add custom filter for JSON parsing in templates
@app.template_filter('from_json')
def from_json_filter(value):
    try:
        return json.loads(value)
    except (ValueError, TypeError):
        return {}

# Configuration
app.config['SECRET_KEY'] = 'car_rental_secret_key_123456'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///car_rental.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Database Models
class Base(DeclarativeBase):
    pass

db = SQLAlchemy(model_class=Base)
db.init_app(app)

class PredictionHistory(db.Model):
    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True, unique=True)
    predicted_price: Mapped[float] = mapped_column(Float)
    vehicle_info: Mapped[str] = mapped_column(String(200))
    prediction_date: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    input_data: Mapped[str] = mapped_column(Text)  # JSON string of input data

# Load ML Model and Data
MODEL_PATH = 'car_rental_model.pkl'
FEATURES_PATH = 'car_rental_model_features.json'
DATA_PATH = 'CarRentalData.csv'

# Load the dataset
try:
    df_data = pd.read_csv(DATA_PATH)
    print(f"Dataset loaded from {DATA_PATH}")
    print(f"Dataset shape: {df_data.shape}")
except FileNotFoundError:
    print(f"Error: Dataset file not found at {DATA_PATH}")
    df_data = None

# Load model with fallback
try:
    ml_model = joblib.load(MODEL_PATH)
    print(f"Model loaded from {MODEL_PATH}")
except Exception as e:
    print(f"Error loading model: {e}")
    # Create fallback model if needed
    if df_data is not None:
        from sklearn.ensemble import RandomForestRegressor
        print("Creating fallback model...")
        fallback_data = df_data.dropna()
        potential_features = ['rating', 'renterTripsTaken', 'reviewCount', 
                            'location.latitude', 'location.longitude', 'vehicle.year']
        available_features = [f for f in potential_features if f in fallback_data.columns]
        
        if available_features:
            X_fallback = fallback_data[available_features]
            y_fallback = fallback_data['rate.daily']
            ml_model = RandomForestRegressor(n_estimators=10, random_state=42)
            ml_model.fit(X_fallback, y_fallback)
            features = available_features
            print("Fallback model created successfully")

# Load features
try:
    with open(FEATURES_PATH, 'r') as f:
        features = json.load(f)
    print(f"Features loaded: {features}")
except FileNotFoundError:
    if 'features' not in locals():
        features = ['rating', 'renterTripsTaken', 'reviewCount', 
                   'location.latitude', 'location.longitude', 'vehicle.year']
    print(f"Using fallback features: {features}")

@app.route('/')
def index():
    """Main dashboard showing recent predictions and data overview"""
    recent_predictions = PredictionHistory.query.order_by(PredictionHistory.prediction_date.desc()).limit(10).all()
    
    # Get CSV data statistics
    csv_stats = {}
    if df_data is not None:
        csv_stats = {
            'total_records': len(df_data),
            'unique_vehicles': len(df_data[['vehicle.make', 'vehicle.model', 'vehicle.year']].drop_duplicates()),
            'unique_locations': len(df_data[['location.city', 'location.state', 'location.country']].drop_duplicates()),
            'avg_daily_rate': df_data['rate.daily'].mean() if 'rate.daily' in df_data.columns else 0,
            'date_range': {
                'min_year': int(df_data['vehicle.year'].min()) if 'vehicle.year' in df_data.columns else 'N/A',
                'max_year': int(df_data['vehicle.year'].max()) if 'vehicle.year' in df_data.columns else 'N/A'
            }
        }
    
    # Get unique values for dropdowns from dataset
    dropdown_data = {}
    if df_data is not None:
        dropdown_data = {
            'fuel_types': sorted(df_data['fuelType'].dropna().unique().tolist()),
            'cities': sorted(df_data['location.city'].dropna().unique().tolist()),
            'countries': sorted(df_data['location.country'].dropna().unique().tolist()),
            'states': sorted(df_data['location.state'].dropna().unique().tolist()),
            'makes': sorted(df_data['vehicle.make'].dropna().unique().tolist()),
            'models': sorted(df_data['vehicle.model'].dropna().unique().tolist()),
            'types': sorted(df_data['vehicle.type'].dropna().unique().tolist()),
            'years': sorted(df_data['vehicle.year'].dropna().unique().tolist(), reverse=True)
        }
    
    return render_template('index.html', 
                         recent_predictions=recent_predictions,
                         csv_stats=csv_stats,
                         dropdown_data=dropdown_data)

@app.route('/predict', methods=['POST'])
def predict():
    """Handle prediction requests"""
    try:
        # Validate that we have the required data
        if df_data is None:
            error_msg = 'Dataset not available for predictions'
            wants_json = 'application/json' in request.headers.get('Accept', '')
            is_ajax = request.headers.get('X-Requested-With') == 'XMLHttpRequest'
            
            if wants_json or is_ajax:
                return jsonify({'success': False, 'error': error_msg}), 400
            else:
                return render_template('prediction_error.html',
                                     error_message="Data source not available",
                                     error_details="The car rental dataset could not be loaded. Please contact support.",
                                     error_technical=error_msg), 400
            
        if ml_model is None:
            error_msg = 'Machine learning model not available'
            wants_json = 'application/json' in request.headers.get('Accept', '')
            is_ajax = request.headers.get('X-Requested-With') == 'XMLHttpRequest'
            
            if wants_json or is_ajax:
                return jsonify({'success': False, 'error': error_msg}), 400
            else:
                return render_template('prediction_error.html',
                                     error_message="Prediction model not available",
                                     error_details="The machine learning model could not be loaded. Please contact support.",
                                     error_technical=error_msg), 400
        
        # Get and validate form data
        try:
            fuel_type = request.form.get('fuelType', '').strip() or 'gasoline'
            rating = float(request.form.get('rating', 4.0))
            renter_trips = int(request.form.get('renterTripsTaken', 1))
            review_count = int(request.form.get('reviewCount', 1))
            city = request.form.get('location.city', '').strip() or 'New York'
            country = request.form.get('location.country', '').strip() or 'United States'
            latitude = float(request.form.get('location.latitude', 40.7128))
            longitude = float(request.form.get('location.longitude', -74.0060))
            state = request.form.get('location.state', '').strip() or 'NY'
            make = request.form.get('vehicle.make', '').strip() or 'Toyota'
            vehicle_model = request.form.get('vehicle.model', '').strip() or 'Camry'
            vehicle_type = request.form.get('vehicle.type', '').strip() or 'car'
            year = int(request.form.get('vehicle.year', 2020))
        except (ValueError, TypeError) as e:
            error_msg = f'Invalid input data: {str(e)}'
            wants_json = 'application/json' in request.headers.get('Accept', '')
            is_ajax = request.headers.get('X-Requested-With') == 'XMLHttpRequest'
            
            if wants_json or is_ajax:
                return jsonify({'success': False, 'error': error_msg}), 400
            else:
                return render_template('prediction_error.html',
                                     error_message="Invalid input values",
                                     error_details="Please check that all numeric fields contain valid numbers.",
                                     error_technical=error_msg), 400
        
        # Prepare input data
        all_input_data = {
            'fuelType': fuel_type,
            'rating': rating,
            'renterTripsTaken': renter_trips,
            'reviewCount': review_count,
            'location.city': city,
            'location.country': country,
            'location.latitude': latitude,
            'location.longitude': longitude,
            'location.state': state,
            'vehicle.make': make,
            'vehicle.model': vehicle_model,
            'vehicle.type': vehicle_type,
            'vehicle.year': year
        }
        
        # Filter for required features
        filtered_input_data = {}
        for feature in features:
            if feature in all_input_data:
                filtered_input_data[feature] = [all_input_data[feature]]
        
        # Ensure we have all required features
        if len(filtered_input_data) == 0:
            error_msg = 'No valid features found for prediction'
            wants_json = 'application/json' in request.headers.get('Accept', '')
            is_ajax = request.headers.get('X-Requested-With') == 'XMLHttpRequest'
            
            if wants_json or is_ajax:
                return jsonify({'success': False, 'error': error_msg}), 400
            else:
                return render_template('prediction_error.html',
                                     error_message="Insufficient data for prediction",
                                     error_details="The provided data doesn't match the model's requirements. Please try different values.",
                                     error_technical=error_msg), 400
        
        input_data = pd.DataFrame(filtered_input_data)
        
        # Handle categorical encoding
        categorical_columns = ['fuelType', 'location.city', 'location.country', 'location.state', 
                             'vehicle.make', 'vehicle.model', 'vehicle.type']
        
        for col in categorical_columns:
            if col in features and col in input_data.columns:
                try:
                    input_value = input_data[col].iloc[0]
                    
                    # Skip if input value is None or empty
                    if input_value is None or str(input_value).strip() == '':
                        continue
                    
                    # Get unique values from dataset
                    if col in df_data.columns:
                        unique_values = df_data[col].dropna().unique()
                        
                        if len(unique_values) == 0:
                            continue
                        
                        # If input value not in dataset, use most common value
                        if input_value not in unique_values:
                            mode_values = df_data[col].mode()
                            most_common = mode_values.iloc[0] if len(mode_values) > 0 else unique_values[0]
                            input_data[col] = most_common
                            input_value = most_common
                        
                        # Encode to numeric
                        unique_sorted = sorted(unique_values)
                        value_to_index = {val: idx for idx, val in enumerate(unique_sorted)}
                        numeric_value = value_to_index.get(input_value, 0)
                        input_data[col] = numeric_value
                    
                except Exception as e:
                    print(f"Warning: Error encoding column {col}: {e}")
                    # Set a default numeric value
                    input_data[col] = 0
        
        # Ensure all required features are present
        missing_features = [f for f in features if f not in input_data.columns]
        if missing_features:
            error_msg = f'Missing required features: {missing_features}'
            wants_json = 'application/json' in request.headers.get('Accept', '')
            is_ajax = request.headers.get('X-Requested-With') == 'XMLHttpRequest'
            
            if wants_json or is_ajax:
                return jsonify({'success': False, 'error': error_msg}), 400
            else:
                return render_template('prediction_error.html',
                                     error_message="Missing required information",
                                     error_details="Some essential data fields are missing. Please fill out the complete form.",
                                     error_technical=error_msg), 400
        
        # Make prediction
        try:
            prediction_input = input_data[features]
            predicted_price = float(ml_model.predict(prediction_input)[0])
            
            # Validate prediction result
            if predicted_price < 0 or predicted_price > 10000:  # Reasonable bounds
                predicted_price = max(10.0, min(predicted_price, 1000.0))  # Clamp to reasonable range
                
        except Exception as e:
            error_msg = f'Prediction failed: {str(e)}'
            wants_json = 'application/json' in request.headers.get('Accept', '')
            is_ajax = request.headers.get('X-Requested-With') == 'XMLHttpRequest'
            
            if wants_json or is_ajax:
                return jsonify({'success': False, 'error': error_msg}), 400
            else:
                return render_template('prediction_error.html',
                                     error_message="Prediction calculation failed",
                                     error_details="The model encountered an error while processing your data. Please try with different values.",
                                     error_technical=error_msg), 400
        
        # Save prediction to database
        vehicle_info = f"{year} {make} {vehicle_model} ({vehicle_type})"
        try:
            prediction = PredictionHistory(
                predicted_price=predicted_price,
                vehicle_info=vehicle_info,
                input_data=json.dumps(all_input_data)
            )
            db.session.add(prediction)
            db.session.commit()
        except Exception as db_error:
            # Log database error but don't fail the prediction
            print(f"Warning: Could not save prediction to database: {db_error}")
            db.session.rollback()
            # Continue with the prediction display even if database save fails
        
        # Generate visualization
        plot_base64 = None
        try:
            plt.figure(figsize=(10, 6))
            
            if df_data is not None and 'vehicle.type' in df_data.columns and 'vehicle.year' in df_data.columns and 'rate.daily' in df_data.columns:
                similar_vehicles = df_data[
                    (df_data['vehicle.type'] == vehicle_type) & 
                    (df_data['vehicle.year'].between(year-2, year+2))
                ]['rate.daily'].dropna()
                
                if len(similar_vehicles) > 0:
                    plt.hist(similar_vehicles, bins=min(20, len(similar_vehicles)), alpha=0.7, color='lightblue', 
                            label=f'Similar {vehicle_type}s ({year-2}-{year+2})')
                    plt.axvline(predicted_price, color='red', linestyle='--', linewidth=2, 
                               label=f'Predicted Price: ${predicted_price:.2f}')
                    plt.xlabel('Daily Rental Rate ($)')
                    plt.ylabel('Number of Vehicles')
                    plt.title(f'Rental Price Comparison for {vehicle_type}s')
                    plt.legend()
                    plt.grid(True, alpha=0.3)
                else:
                    plt.text(0.5, 0.5, 'No similar vehicles found for comparison', 
                            ha='center', va='center', transform=plt.gca().transAxes, fontsize=14)
                    plt.title(f'Predicted Price: ${predicted_price:.2f}')
            else:
                plt.text(0.5, 0.5, f'Predicted Price: ${predicted_price:.2f}', 
                        ha='center', va='center', transform=plt.gca().transAxes, fontsize=16)
                plt.title('Car Rental Price Prediction')
            
            plt.tight_layout()
            
            # Save plot to base64
            img_bytes = io.BytesIO()
            plt.savefig(img_bytes, format='png', dpi=100, bbox_inches='tight')
            img_bytes.seek(0)
            plot_base64 = base64.b64encode(img_bytes.read()).decode('utf-8')
            plt.close()
            
        except Exception as e:
            print(f"Warning: Could not generate plot: {e}")
            try:
                plt.close()  # Ensure matplotlib cleanup
            except:
                pass
        
        flash(f'Prediction successful! Estimated daily rate: ${predicted_price:.2f}', 'success')
        
        response_data = {
            'success': True,
            'predicted_price': f'{predicted_price:.2f}',
            'vehicle_info': vehicle_info
        }
        
        # Only include plot if it was generated successfully
        if plot_base64:
            response_data['plot_image'] = plot_base64
        
        # Check if request wants HTML response (from form) or JSON (from AJAX)
        wants_json = 'application/json' in request.headers.get('Accept', '')
        is_ajax = request.headers.get('X-Requested-With') == 'XMLHttpRequest'
        full_page_requested = request.form.get('fullPageResult') is not None
        
        if (not wants_json and not is_ajax) or full_page_requested:
            # Render HTML template for direct form submissions or when full page is requested
            return render_template('prediction_result.html',
                                 predicted_price=f'{predicted_price:.2f}',
                                 vehicle_info=vehicle_info,
                                 plot_image=plot_base64,
                                 input_data=all_input_data,
                                 prediction_date=datetime.utcnow())
        else:
            # Return JSON for AJAX requests
            return jsonify(response_data)
        
    except Exception as e:
        print(f"Prediction error: {str(e)}")  # Log for debugging
        try:
            plt.close('all')  # Clean up any matplotlib figures
        except:
            pass
        
        # Collect input data for error template
        error_input_data = {}
        try:
            error_input_data = {
                'fuelType': request.form.get('fuelType', ''),
                'rating': request.form.get('rating', ''),
                'renterTripsTaken': request.form.get('renterTripsTaken', ''),
                'reviewCount': request.form.get('reviewCount', ''),
                'location.city': request.form.get('location.city', ''),
                'location.country': request.form.get('location.country', ''),
                'location.latitude': request.form.get('location.latitude', ''),
                'location.longitude': request.form.get('location.longitude', ''),
                'location.state': request.form.get('location.state', ''),
                'vehicle.make': request.form.get('vehicle.make', ''),
                'vehicle.model': request.form.get('vehicle.model', ''),
                'vehicle.type': request.form.get('vehicle.type', ''),
                'vehicle.year': request.form.get('vehicle.year', '')
            }
        except:
            pass
        
        # Check if request wants HTML response (from form) or JSON (from AJAX)
        wants_json = 'application/json' in request.headers.get('Accept', '')
        is_ajax = request.headers.get('X-Requested-With') == 'XMLHttpRequest'
        
        if wants_json or is_ajax:
            # Return JSON for AJAX requests
            return jsonify({
                'success': False,
                'error': str(e)
            }), 400
        else:
            # Render error template for form submissions
            error_message = "Prediction failed due to an unexpected error."
            error_details = str(e)
            
            # Categorize common errors for better user experience
            if "sqlite" in str(e).lower() or "database" in str(e).lower():
                error_message = "Database error occurred while saving prediction."
                error_details = "The prediction may have been calculated but couldn't be saved. Please try again."
            elif "model" in str(e).lower() or "predict" in str(e).lower():
                error_message = "Machine learning model error."
                error_details = "The model couldn't process your input data. Please check your values and try again."
            elif "feature" in str(e).lower() or "missing" in str(e).lower():
                error_message = "Input data validation error."
                error_details = "Some required information is missing or invalid. Please fill all fields correctly."
            
            return render_template('prediction_error.html',
                                 error_message=error_message,
                                 error_details=error_details,
                                 error_technical=str(e),
                                 input_data=error_input_data), 400

@app.route('/prediction/<int:prediction_id>')
def view_prediction(prediction_id):
    """View a specific prediction result"""
    prediction = PredictionHistory.query.get_or_404(prediction_id)
    
    # Parse the input data JSON
    try:
        input_data = json.loads(prediction.input_data)
    except (json.JSONDecodeError, TypeError):
        input_data = {}
    
    # Try to regenerate the plot if possible
    plot_base64 = None
    try:
        if df_data is not None:
            vehicle_type = input_data.get('vehicle.type', 'car')
            year = int(input_data.get('vehicle.year', 2020))
            
            plt.figure(figsize=(10, 6))
            
            similar_vehicles = df_data[
                (df_data['vehicle.type'] == vehicle_type) & 
                (df_data['vehicle.year'].between(year-2, year+2))
            ]['rate.daily'].dropna()
            
            if len(similar_vehicles) > 0:
                plt.hist(similar_vehicles, bins=min(20, len(similar_vehicles)), alpha=0.7, color='lightblue',
                        label=f'Similar {vehicle_type}s ({year-2}-{year+2})')
                plt.axvline(prediction.predicted_price, color='red', linestyle='--', linewidth=2,
                           label=f'Predicted Price: ${prediction.predicted_price:.2f}')
                plt.xlabel('Daily Rental Rate ($)')
                plt.ylabel('Number of Vehicles')
                plt.title(f'Rental Price Comparison for {vehicle_type}s')
                plt.legend()
                plt.grid(True, alpha=0.3)
            else:
                plt.text(0.5, 0.5, f'Predicted Price: ${prediction.predicted_price:.2f}',
                        ha='center', va='center', transform=plt.gca().transAxes, fontsize=16)
                plt.title('Car Rental Price Prediction')
            
            plt.tight_layout()
            
            img_bytes = io.BytesIO()
            plt.savefig(img_bytes, format='png', dpi=100, bbox_inches='tight')
            img_bytes.seek(0)
            plot_base64 = base64.b64encode(img_bytes.read()).decode('utf-8')
            plt.close()
    except Exception as e:
        print(f"Could not regenerate plot: {e}")
        try:
            plt.close()
        except:
            pass
    
    return render_template('prediction_result.html',
                         predicted_price=f'{prediction.predicted_price:.2f}',
                         vehicle_info=prediction.vehicle_info,
                         plot_image=plot_base64,
                         input_data=input_data,
                         prediction_date=prediction.prediction_date)

@app.route('/history')
def prediction_history():
    """Show prediction history"""
    predictions = PredictionHistory.query.order_by(PredictionHistory.prediction_date.desc()).all()
    return render_template('history.html', predictions=predictions)

@app.route('/data_overview')
def data_overview():
    """Show overview of available data from CSV"""
    if df_data is None:
        flash('No data file available', 'error')
        return redirect(url_for('index'))
    
    # Calculate comprehensive data statistics
    stats = {
        'total_records': len(df_data),
        'vehicles': {
            'total_unique': len(df_data[['vehicle.make', 'vehicle.model', 'vehicle.year']].drop_duplicates()),
            'makes': len(df_data['vehicle.make'].unique()),
            'models': len(df_data['vehicle.model'].unique()),
            'year_range': f"{int(df_data['vehicle.year'].min())}-{int(df_data['vehicle.year'].max())}"
        },
        'locations': {
            'total_unique': len(df_data[['location.city', 'location.state', 'location.country']].drop_duplicates()),
            'cities': len(df_data['location.city'].unique()),
            'states': len(df_data['location.state'].unique()),
            'countries': len(df_data['location.country'].unique())
        },
        'pricing': {
            'avg_daily_rate': f"${df_data['rate.daily'].mean():.2f}",
            'min_daily_rate': f"${df_data['rate.daily'].min():.2f}",
            'max_daily_rate': f"${df_data['rate.daily'].max():.2f}",
            'median_daily_rate': f"${df_data['rate.daily'].median():.2f}"
        },
        'ratings': {
            'avg_rating': f"{df_data['rating'].mean():.2f}",
            'total_reviews': int(df_data['reviewCount'].sum()),
            'avg_trips': f"{df_data['renterTripsTaken'].mean():.1f}"
        }
    }
    
    return jsonify({
        'success': True,
        'data_stats': stats,
        'message': 'Data loaded from CSV file - no database storage needed for vehicle/location data'
    })

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(debug=True)