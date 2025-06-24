from flask import Flask, request, jsonify
from flask_sqlalchemy import SQLAlchemy
from flask_cors import CORS
from datetime import datetime, timedelta
import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import pickle
import os
import json

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///tasks.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)
CORS(app)

# Database Models
class Task(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.String(200), nullable=False)
    description = db.Column(db.Text)
    difficulty = db.Column(db.Integer, nullable=False)  # 1-5 scale
    importance = db.Column(db.Integer, nullable=False)  # 1-5 scale
    estimated_duration = db.Column(db.Integer, nullable=False)  # minutes
    actual_duration = db.Column(db.Integer)  # minutes
    energy_required = db.Column(db.Integer, nullable=False)  # 1-5 scale
    scheduled_time = db.Column(db.DateTime)
    completed_time = db.Column(db.DateTime)
    is_completed = db.Column(db.Boolean, default=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

class EnergyLog(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    timestamp = db.Column(db.DateTime, nullable=False)
    energy_level = db.Column(db.Integer, nullable=False)  # 1-5 scale
    hour_of_day = db.Column(db.Integer, nullable=False)
    day_of_week = db.Column(db.Integer, nullable=False)
    tasks_completed = db.Column(db.Integer, default=0)

# ML Model for Energy Prediction
class EnergyPredictor:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.is_trained = False
        
    def prepare_features(self, hour, day_of_week, recent_completions=0):
        """Prepare features for prediction"""
        return np.array([[
            hour,
            day_of_week,
            recent_completions,
            np.sin(2 * np.pi * hour / 24),  # Cyclical hour encoding
            np.cos(2 * np.pi * hour / 24),
            np.sin(2 * np.pi * day_of_week / 7),  # Cyclical day encoding
            np.cos(2 * np.pi * day_of_week / 7)
        ]])
    
    def train_model(self):
        """Train energy prediction model"""
        # Get training data
        energy_logs = EnergyLog.query.all()
        if len(energy_logs) < 10:
            return False
            
        X = []
        y = []
        
        for log in energy_logs:
            features = self.prepare_features(
                log.hour_of_day, 
                log.day_of_week, 
                log.tasks_completed
            )
            X.append(features[0])
            y.append(log.energy_level)
        
        X = np.array(X)
        y = np.array(y)
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Build TensorFlow model
        self.model = tf.keras.Sequential([
            tf.keras.layers.Dense(64, activation='relu', input_shape=(7,)),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(16, activation='relu'),
            tf.keras.layers.Dense(1, activation='sigmoid')  # Output 0-1, scale to 1-5
        ])
        
        self.model.compile(
            optimizer='adam',
            loss='mse',
            metrics=['mae']
        )
        
        # Train model
        self.model.fit(
            X_scaled, y,
            epochs=100,
            batch_size=8,
            validation_split=0.2,
            verbose=0
        )
        
        self.is_trained = True
        return True
    
    def predict_energy(self, hour, day_of_week, recent_completions=0):
        """Predict energy level for given time"""
        if not self.is_trained:
            return 3  # Default medium energy
            
        features = self.prepare_features(hour, day_of_week, recent_completions)
        features_scaled = self.scaler.transform(features)
        
        prediction = self.model.predict(features_scaled, verbose=0)[0][0]
        # Scale from 0-1 to 1-5
        return max(1, min(5, int(prediction * 4 + 1)))

# Initialize ML model
energy_predictor = EnergyPredictor()

# Task Scheduling Algorithm
class TaskScheduler:
    @staticmethod
    def calculate_priority_score(task, predicted_energy):
        """Calculate task priority based on multiple factors"""
        # Base score from importance and difficulty
        urgency_score = task.importance * 0.4
        difficulty_score = (6 - task.difficulty) * 0.2  # Easier tasks get higher score when energy is low
        
        # Energy matching - high energy tasks need high energy levels
        energy_match = 1.0
        if task.energy_required > predicted_energy:
            energy_match = 0.5  # Penalty for scheduling high-energy task when energy is low
        elif task.energy_required < predicted_energy - 1:
            energy_match = 0.8  # Slight penalty for using high energy on low-energy tasks
            
        return (urgency_score + difficulty_score) * energy_match
    
    @staticmethod
    def schedule_tasks():
        """Schedule all unfinished tasks based on energy predictions"""
        unscheduled_tasks = Task.query.filter_by(is_completed=False, scheduled_time=None).all()
        
        if not unscheduled_tasks:
            return []
            
        scheduled_tasks = []
        current_time = datetime.now().replace(minute=0, second=0, microsecond=0)
        
        # Schedule tasks for next 7 days
        for day in range(7):
            for hour in range(8, 22):  # Working hours 8 AM to 10 PM
                schedule_time = current_time + timedelta(days=day, hours=hour-current_time.hour)
                
                # Predict energy for this time slot
                predicted_energy = energy_predictor.predict_energy(
                    hour, 
                    schedule_time.weekday(),
                    0  # Will be updated based on actual scheduling
                )
                
                # Find best task for this time slot
                best_task = None
                best_score = 0
                
                for task in unscheduled_tasks:
                    if task in scheduled_tasks:
                        continue
                        
                    score = TaskScheduler.calculate_priority_score(task, predicted_energy)
                    
                    if score > best_score:
                        best_score = score
                        best_task = task
                
                if best_task:
                    best_task.scheduled_time = schedule_time
                    scheduled_tasks.append(best_task)
                    
                    # Remove from unscheduled list
                    unscheduled_tasks.remove(best_task)
                    
                    if not unscheduled_tasks:
                        break
            
            if not unscheduled_tasks:
                break
        
        # Save scheduled tasks
        db.session.commit()
        return scheduled_tasks

# API Routes
@app.route('/api/tasks', methods=['GET'])
def get_tasks():
    tasks = Task.query.all()
    return jsonify([{
        'id': task.id,
        'title': task.title,
        'description': task.description,
        'difficulty': task.difficulty,
        'importance': task.importance,
        'estimated_duration': task.estimated_duration,
        'actual_duration': task.actual_duration,
        'energy_required': task.energy_required,
        'scheduled_time': task.scheduled_time.isoformat() if task.scheduled_time else None,
        'completed_time': task.completed_time.isoformat() if task.completed_time else None,
        'is_completed': task.is_completed,
        'created_at': task.created_at.isoformat()
    } for task in tasks])

@app.route('/api/tasks', methods=['POST'])
def create_task():
    data = request.json
    
    task = Task(
        title=data['title'],
        description=data.get('description', ''),
        difficulty=data['difficulty'],
        importance=data['importance'],
        estimated_duration=data['estimated_duration'],
        energy_required=data['energy_required']
    )
    
    db.session.add(task)
    db.session.commit()
    
    # Trigger rescheduling
    TaskScheduler.schedule_tasks()
    
    return jsonify({'message': 'Task created successfully', 'id': task.id}), 201

@app.route('/api/tasks/<int:task_id>/complete', methods=['POST'])
def complete_task(task_id):
    task = Task.query.get_or_404(task_id)
    data = request.json
    
    task.is_completed = True
    task.completed_time = datetime.utcnow()
    task.actual_duration = data.get('actual_duration', task.estimated_duration)
    
    db.session.commit()
    
    # Log energy data point
    current_hour = datetime.now().hour
    energy_log = EnergyLog(
        timestamp=datetime.utcnow(),
        energy_level=data.get('energy_level', 3),
        hour_of_day=current_hour,
        day_of_week=datetime.now().weekday(),
        tasks_completed=1
    )
    
    db.session.add(energy_log)
    db.session.commit()
    
    # Retrain model periodically
    if EnergyLog.query.count() % 10 == 0:
        energy_predictor.train_model()
    
    return jsonify({'message': 'Task completed successfully'})

@app.route('/api/energy/log', methods=['POST'])
def log_energy():
    """Manual energy level logging"""
    data = request.json
    
    energy_log = EnergyLog(
        timestamp=datetime.utcnow(),
        energy_level=data['energy_level'],
        hour_of_day=datetime.now().hour,
        day_of_week=datetime.now().weekday(),
        tasks_completed=0
    )
    
    db.session.add(energy_log)
    db.session.commit()
    
    return jsonify({'message': 'Energy logged successfully'})

@app.route('/api/schedule', methods=['GET'])
def get_schedule():
    """Get scheduled tasks for today"""
    today_start = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    today_end = today_start + timedelta(days=1)
    
    scheduled_tasks = Task.query.filter(
        Task.scheduled_time >= today_start,
        Task.scheduled_time < today_end,
        Task.is_completed == False
    ).order_by(Task.scheduled_time).all()
    
    return jsonify([{
        'id': task.id,
        'title': task.title,
        'scheduled_time': task.scheduled_time.isoformat(),
        'estimated_duration': task.estimated_duration,
        'energy_required': task.energy_required,
        'difficulty': task.difficulty,
        'importance': task.importance
    } for task in scheduled_tasks])

@app.route('/api/energy/predict', methods=['GET'])
def predict_energy():
    """Predict energy for next few hours"""
    current_time = datetime.now()
    predictions = []
    
    for i in range(8):  # Next 8 hours
        future_time = current_time + timedelta(hours=i)
        predicted_energy = energy_predictor.predict_energy(
            future_time.hour,
            future_time.weekday()
        )
        
        predictions.append({
            'time': future_time.isoformat(),
            'predicted_energy': predicted_energy
        })
    
    return jsonify(predictions)

@app.route('/api/reschedule', methods=['POST'])
def reschedule_tasks():
    """Manually trigger task rescheduling"""
    scheduled_tasks = TaskScheduler.schedule_tasks()
    
    return jsonify({
        'message': f'Rescheduled {len(scheduled_tasks)} tasks',
        'scheduled_count': len(scheduled_tasks)
    })

# Initialize database
def create_tables():
    with app.app_context():
        db.create_all()
        
        # Add sample data if database is empty
        if Task.query.count() == 0:
            sample_tasks = [
                Task(title="Code ML model", difficulty=5, importance=4, estimated_duration=120, energy_required=5),
                Task(title="Write documentation", difficulty=2, importance=3, estimated_duration=60, energy_required=2),
                Task(title="Team meeting", difficulty=1, importance=4, estimated_duration=30, energy_required=3),
                Task(title="Review code", difficulty=3, importance=3, estimated_duration=45, energy_required=4)
            ]
            
            for task in sample_tasks:
                db.session.add(task)
            
            db.session.commit()

if __name__ == '__main__':
    # Initialize database before running app
    create_tables()
    app.run(debug=True, host='0.0.0.0', port=5000)