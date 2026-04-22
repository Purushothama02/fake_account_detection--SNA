# Fake Account Detection in Social Networks using Machine Learning

## M.Tech Mini Project – Social Network Analysis (SNA)

### Project Overview

This project focuses on detecting fake social media accounts using Machine Learning techniques combined with Social Network Analysis (SNA) concepts.

Fake accounts, bots, and spam profiles are major problems on social media platforms such as Twitter, Instagram, and Facebook. This project uses user profile features and behavioral attributes to classify whether an account is real or fake.

The system compares multiple machine learning models and identifies the best-performing model for accurate fake account detection.

---

## Dataset Used

### Fake Social Media Account Detection Dataset

The dataset contains 3000 labeled social media profiles with multiple features such as:

* platform
* has_profile_pic
* bio_length
* username_randomness
* followers
* following
* follower_following_ratio
* account_age_days
* posts
* posts_per_day
* caption_similarity_score
* content_similarity_score
* follow_unfollow_rate
* spam_comments_rate
* generic_comment_rate
* suspicious_links_in_bio
* verified
* is_fake (Target Variable)

---

## Machine Learning Models Used

The following models were implemented and compared:

1. Logistic Regression
2. Decision Tree Classifier
3. Random Forest Classifier

---

## Best Model

### Random Forest Classifier

Random Forest achieved the highest accuracy and provided better performance compared to other models because it handles nonlinear relationships and feature interactions effectively.

---

## Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn
* NetworkX
* Jupyter Notebook / VS Code

---

## Project Structure

fake-account-detection-sna/

│

├── model.py

├── fake_social_media.csv

├── README.md

├── requirements.txt

├── report.pdf

└── presentation.pptx

---

## Installation

Install required dependencies using:

pip install pandas numpy matplotlib seaborn scikit-learn networkx xgboost jupyter

---

## Run the Project

Execute the Python file using:

python model.py

---

## Output

The project generates:

* Accuracy comparison of all models
* Classification reports
* Confusion Matrix
* Accuracy comparison graph
* Final best model selection

---

## Conclusion

Fake account detection is an important problem in modern social networks.

This project successfully applies Machine Learning techniques for classification and demonstrates that Random Forest is the most effective model for this use case.

The project also shows how Social Network Analysis can improve cybersecurity and platform trust.

---

## Author
Purushothama N
M.Tech Student
Mini Project – Social Network Analysis (SNA)

---
