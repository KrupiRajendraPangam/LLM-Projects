## Setup Instructions

### 1. Create a Conda Environment or Python virtual environment

Run the following command to create a new Conda environment:

```bash
conda create -p myvenv python=3.11 -y
```

### 2. Activate the Environment

Activate the newly created environment:

```
conda activate ./myvenv
```

### 3. Install Dependencies

Install all required Python packages using pip:

 ```
pip install -r requirements.txt
```

### 4. Run the Streamlit App

Start the Streamlit application with:

 ```
streamlit run app.py
```