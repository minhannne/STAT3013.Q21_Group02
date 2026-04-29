# The Impact of Pricing on Consumer Purchase Decisions

## 1. Project Title

- **Course:** STAT3013.Q21.CTTT  
- **Group:** STAT3013.Q21_Group02  
- **Topic:** Analyzing the impact of pricing on consumer purchase decisions using Machine Learning and Deep Learning. This project implements four distinct models including ElasticNet, LightGBM, TabNet, and Wide & Deep to predict purchase Quantity. By comparing model metrics and analyzing Feature Importance, we identify the key factors influencing consumer behavior and the specific role of price in the decision-making process.
---

## 2. Run Instructions
To run this project on your local machine, follow these steps:
- 1. **Clone the repository:**
   ```bash
git clone https://github.com/minhannne/STAT3013.Q21_Group02
cd STAT3013.Q21_Group02
- 2. **Install required libraries:**
pip install -r requirements.txt
- 3. **Execution order:**
   - Ensure the dataset is downloaded and placed in the `data/` folder.
   - Run individual models in the `src/` folder (e.g., `python src/LightGBM.py`).
   - Run visualization scripts in `visualization/` to see comparative results.

---

## 3. Environment Requirements
- **Python Version:** 3.9 or higher
- **Core Libraries:** `Pandas`, `Numpy` (Data Manipulation)
- **Visualization:** `Matplotlib`, `Seaborn` (Data Visualization)
- **Machine Learning & Deep Learning:** - `TensorFlow` (for Wide & Deep)
  - `PyTorch` (for TabNet)
  - `LightGBM` (Gradient Boosting)
  - `Scikit-learn` (ElasticNet & Evaluation Metrics)
  
---

## 4. Dataset & Video Demo Links
- **Dataset:** [Kaggle - Dunnhumby The Complete Journey](https://www.kaggle.com/datasets/frtgnn/dunnhumby-the-complete-journey/data)
- **Note on Data Scaling:** 
    - **Complex Merging:** The raw dataset consists of multiple relational tables (including `transaction_data`, `product`, `hh_demographic`, `campaign_table`, etc.). We performed extensive data joining and cleaning to consolidate these files into a unified dataset for analysis.
    - **Sampling for Performance:** Due to local hardware constraints and the large size of the merged results, the models were trained and evaluated on the **first 500,000 records** (via `df.head(500000)`). This approach ensures computational stability while still providing a robust sample size for deep learning models like TabNet and Wide & Deep.
- **Demo Video:** [Click here to watch on Drive](https://drive.google.com/file/d/1nmwQpuqJVE1h-3UmtTehaL7HA5BxJY1U/view?usp=drive_link)
---

## 5. License
This project is licensed under the **MIT License**.
