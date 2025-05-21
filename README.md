# **Project Setup and Deployment: SolarPower-Forecast**

These instructions will guide you through setting up the project environment, preparing the data, training the necessary model, and running the Streamlit application.

## **1. Prerequisites**
**Git**: You'll need Git installed to clone the repository. You can download it from https://git-scm.com/.

**Python**: Python 3.8 or newer is recommended. You can download it from https://www.python.org/. Pip (Python package installer) will be included with your Python installation.

## **2. Environment Setup and Installation**
### **a. Clone the Repository**
Open your terminal or command prompt and run the following commands:
```
git clone [https://github.com/Nawfel-9/SolarPower-Forecast.git](https://github.com/Nawfel-9/SolarPower-Forecast.git)
cd SolarPower-Forecast
```

### **b. Create and Activate a Virtual Environment**
It's highly recommended to use a virtual environment to manage project dependencies and avoid conflicts.

Create the virtual environment (e.g., named venv):
```
python -m venv venv
```
(Alternatively, if you prefer Conda, you can create an environment with a command like: `conda create -n solarforecast_env python=3.8 -y`)

Activate the virtual environment:

- Windows:
```
venv\Scripts\activate
```
- Linux/macOS:
```
source venv/bin/activate
```
(If using Conda, activate with: `conda activate solarforecast_env`)

You should see the virtual environment's name (e.g., (venv) or (solarforecast_env)) at the beginning of your terminal prompt.

### **c. Install Required Libraries**
With your virtual environment activated, install the necessary Python packages.
First, install PyTorch (the command below specifies the CPU version for broader compatibility or check requirements.txt if you want to use gpu):
```
pip install --no-cache-dir torch torchvision torchaudio --index-url [https://download.pytorch.org/whl/cpu](https://download.pytorch.org/whl/cpu)
```

Then, install the remaining project dependencies listed in requirements.txt:
```
pip install -r requirements.txt
```


## **3. Data Preparation and Model Training (To Recreate Results)**
Follow these steps if you wish to recreate the dataset from scratch and train the SARIMA model. Note that a pre-trained LSTM model checkpoint is already included in the repository, so this section might be optional if you only want to run the application with existing models.
### **a. Download the Dataset**
Download the dataset from Kaggle: [Solar Power Generation and Consumption Dataset](https://kaggle.com/datasets/77683f114a97ab3ad9f7cfd138528bb1269836a29e085c56e24190f140d3303a)
(A Kaggle account may be required for downloading.)
Extract the downloaded files and place them into the data/ directory within your SolarPower-Forecast project folder. Create the data/ directory if it doesn't exist.
### **b. Run Preprocessing Scripts**
These scripts will process the raw data into a format suitable for model training and application use.
`python consumed_cost_energy_data.py`
`python generated_energy_estimation.py`


**IMPORTANT**: Before running these scripts, you must open consumed_cost_energy_data.py and generated_energy_estimation.py in a text editor. Carefully review and update any internal file paths within these scripts to correctly point to your input data (downloaded in step 3a) and your desired output locations for the processed data.
### **c. Train the SARIMA Model**
After the data has been successfully processed, run the SARIMA training script. This step typically only needs to be performed once.
```
python train/train_sarima.py
```



(As mentioned, the LSTM model has a pre-loaded checkpoint available in the repository, so retraining it might not be necessary to run the application.)
## **4. Running the Application**
Once the setup is complete and, if necessary, the data preparation and model training steps have been performed, you can run the Streamlit web application.
Ensure your virtual environment is still active.
In your terminal, make sure you are in the project's root directory (SolarPower-Forecast).
Launch the Streamlit app using the following command:
```
streamlit run app.py
```
This command will typically start a local web server and open the application in your default web browser.
Troubleshooting Note: If the streamlit run app.py command results in an error like "'streamlit' is not recognized..." or "command not found: streamlit", it might indicate an issue with your system's PATH or the virtual environment. In such cases, try running Streamlit as a Python module:
```
python -m streamlit run app.py
```

## **5. Stopping the Application**
To stop the Streamlit application, go back to the terminal window where it's running and press Ctrl+C.
