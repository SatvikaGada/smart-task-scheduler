# Smart Task Scheduler - Setup Guide

## What This App Does
This is an AI-powered task scheduler that learns your energy patterns and optimally schedules tasks based on:
- Your predicted energy levels throughout the day  
- Task difficulty, importance, and energy requirements  
- Historical completion patterns  
- Time-of-day and day-of-week patterns  

## Key Features
- **Energy Pattern Learning**: TensorFlow neural network predicts your energy levels  
- **Smart Scheduling**: Matches high-energy tasks with high-energy time slots  
- **Adaptive Algorithms**: Gets better over time as you use it  
- **Real-time Optimization**: Reschedules tasks based on completion patterns  

## Requirements

### Backend Requirements (`requirements.txt`)
Flask==2.3.3
Flask-SQLAlchemy==3.0.5
Flask-CORS==4.0.0
tensorflow==2.13.0
numpy==1.24.3
pandas==2.0.3
scikit-learn==1.3.0


### System Requirements
- Python 3.8+  
- Node.js (for mobile app deployment)  
- 4GB RAM minimum (for TensorFlow)  

## Installation Steps

### 1. Backend Setup
```bash
# Create project directory
mkdir smart-task-scheduler
cd smart-task-scheduler

# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate

# Install dependencies
pip install Flask==2.3.3 Flask-SQLAlchemy==3.0.5 Flask-CORS==4.0.0 tensorflow==2.13.0 numpy==1.24.3 pandas==2.0.3 scikit-learn==1.3.0

# Copy the backend code into app.py

# Run backend
python app.py
2. Frontend Setup
# Copy the frontend code into index.html

# Simply open index.html in your browser
File Structure
smart-task-scheduler/
├── app.py                 # Backend Flask application
├── index.html            # Frontend React application
├── requirements.txt      # Python dependencies
├── tasks.db              # SQLite database 
└── README.md             
How to Use
1. Initial Setup
Start the backend server: python app.py

Open index.html in your browser

Create a few tasks with different difficulty/importance levels

Log your energy levels periodically

2. Training the AI
Use the app for 1-2 weeks regularly

Log your energy levels at different times

Complete tasks and rate your energy when completing

The AI will start making better predictions after ~50 data points

3. Key Concepts to Understand
Energy Levels (1-5):

1: Exhausted, can barely think

2: Low energy, good for simple tasks

3: Normal energy, moderate tasks

4: High energy, complex tasks

5: Peak energy, most challenging work

Task Difficulty (1-5):

1: Simple, routine tasks

2: Easy tasks requiring minimal thought

3: Moderate complexity

4: Complex, requires focus

5: Most challenging, creative work

Scheduling Algorithm:

Matches high-energy tasks with predicted high-energy times

Considers task importance and deadlines

Optimizes for maximum productivity

The AI/ML Components
1. Energy Prediction Model (TensorFlow)
Neural network architecture:

Input: Hour, day_of_week, recent_completions, cyclical_encodings

Hidden layers: 64 → 32 → 16 neurons with ReLU activation

Output: Energy level prediction (1-5)

Training: MSE loss with Adam optimizer

2. Task Scheduling Algorithm
Priority scoring formula:

ini
Copy
Edit
priority = (importance * 0.4 + difficulty_factor * 0.2) * energy_match_factor
Energy matching:

High energy tasks get penalty if scheduled during low energy

Optimal matching increases task completion probability

3. Continuous Learning
Retrains model every 10 completed tasks

Updates predictions based on actual vs predicted performance

Adapts to changing patterns over time
