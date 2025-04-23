# 🏠 House Price Predictor

A machine learning project to predict house prices based on various features using regression models.

## 📁 Project Structure

 House-Price-Predictor/ 
 ├── .venv/ # Virtual environment 
 ├── data/ # Raw and processed data 
 ├── models/ # Saved trained models 
 ├── notebooks/ # Jupyter notebooks for exploration 
 ├── reports/ # Evaluation reports, plots, metrics 
 ├── src/ # Source code 
 │ ├── data_loader.py # Data loading and preprocessing 
 │ ├── model.py # Model building 
 │ ├── predict.py # Prediction logic 
 │ ├── train.py # Training pipeline 
 │ └── utils.py # Utility functions 
 ├── .gitignore # Files and folders to ignore in Git 
 ├── house-price-predictor.code-workspace # VSCode workspace settings 
 ├── main.py # Main script to run the project 
 ├── README.md # Project documentation 
 └── requirements.txt # Python dependencies


## 🚀 How to Run

1. **Clone the repo**
   ```bash
   git clone https://github.com/your-username/House-Price-Predictor.git
   cd House-Price-Predictor

2.   **Create virtual environment**
    python -m venv .venv
    .venv\Scripts\activate    # Windows

3. **Install dependencies**
    pip install -r requirements.txt

4. **Run the pipeline**
    python main.py

## Requirements
Python 3.8+
pandas, scikit-learn, matplotlib, seaborn, etc.

## Dataset
The dataset used for this project contains various features such as the number of bedrooms, square footage, main road accessiblity, and other relevant metrics that influence house prices. It is available in CSV format in the `data/` directory.

## License
This project is open source and available under the MIT License.