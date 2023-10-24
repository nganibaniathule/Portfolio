# Bank Customer Churn Prediction Project
## Introduction

In an era of fierce competition in the banking industry, retaining customers has become a paramount concern for financial institutions. The phenomenon of customer churn, the process by which customers cease their association with a bank, can have significant repercussions on a bank's profitability and overall performance. The ability to predict and mitigate customer churn is, therefore, a strategic imperative for any modern bank.

Bank Customer Churn Prediction is a critical application of data science and machine learning techniques to forecast which customers are likely to leave a bank in the near future. This predictive modeling not only serves as a proactive measure to retain customers but also as an opportunity to enhance customer satisfaction and experience.

The goal of this project is to develop a robust predictive model that identifies at-risk customers by analyzing various features such as transaction history, demographics, account activity, and customer interactions. By understanding the underlying patterns and behaviors of customers who have previously churned, banks can implement targeted retention strategies, offer personalized incentives, and ultimately reduce the attrition rate.

This portfolio project delves into the realm of bank customer churn prediction, showcasing the application of data preprocessing, feature engineering, and machine learning algorithms to build a predictive model. Through this project, we explore the significance of data-driven decision-making in the financial sector and demonstrate the potential impact of predictive analytics on customer retention and revenue growth.

Join me on this journey to harness the power of data science to address one of the most pressing challenges faced by the banking industry: predicting and preventing customer churn.


## Problem Statement:

"In the dynamic landscape of the banking industry, customer churn, or the loss of valued clients, has emerged as a pressing concern for financial institutions. As a proactive response to this challenge, our project aims to develop an effective predictive model for bank customer churn. The primary objective is to identify and predict potential churners among the bank's existing customer base, allowing the bank to take targeted actions to retain these customers.

The project entails the utilization of data science and machine learning techniques to create a predictive model that can accurately flag individuals at risk of churning. By analyzing a diverse set of customer-related data, including transaction history, demographics, account activity, and customer interactions, we seek to uncover underlying patterns, behaviors, and factors contributing to churn.

The key questions we aim to address are:
- Can we accurately predict which customers are likely to churn in the near future?
* What features and factors are significant predictors of churn?
+ How can this predictive model be leveraged to implement retention strategies and improve overall customer satisfaction?

Our ultimate goal is to equip banks with a predictive tool that aids in the identification of customers at risk of leaving, thereby empowering them to deploy personalized engagement and retention initiatives. Through the development of this model, we aspire to not only reduce customer churn but also enhance customer relationships and financial performance in the banking sector."

## Data Collection:

The dataset used in this project was sourced from Kaggle, a renowned platform for data science and machine learning resources. It is essential to note that the dataset was already in a well-prepared and clean state, with no missing values, making it readily usable for analysis and model development. The data was provided in the form of a CSV (Comma-Separated Values) file, a common and easily accessible data format in the Python programming environment.

This streamlined data collection process facilitated efficient data exploration, preprocessing, and the development of a robust predictive model. The CSV format's compatibility with Python allowed for seamless integration into the project, enabling straightforward data manipulation, visualization, and machine learning model building.

## Data Preprocessing:

The dataset used in this project was collected from Kaggle and was exceptionally well-prepared, requiring minimal initial data cleaning. Notably, there were no missing or null values in the dataset, ensuring the integrity of the information used for analysis and modeling.

To harness the predictive power of non-numeric features and enable their inclusion in our models, we embarked on an encoding process. Non-numeric columns were converted into categorical variables, allowing us to integrate these essential aspects of the data into our analysis effectively.

However, the dataset presented a challenge in the form of numeric columns with varying ranges. This diversity in scale among numerical attributes can potentially hinder the performance of machine learning models. To address this issue, we applied the MinMaxScaler, a transformation technique provided by the Scikit-Learn library. This scaler normalized the numeric columns, ensuring that all features were on the same scale within a specified range. This preprocessing step was crucial in enhancing the model's ability to effectively interpret and weigh each feature, ultimately contributing to more accurate predictions.

The combination of encoding non-numeric columns and scaling numerical ones laid a solid foundation for our subsequent analysis and predictive modeling efforts.


## Exploratory Data Analysis (EDA):

During the exploratory phase of our analysis, we gained valuable insights into the dataset, shedding light on patterns and trends that form the foundation of our Bank Customer Churn Prediction project.

- Churn Distribution:
   - Our dataset comprises 7,963 non-churners and 2,037 churners represented by 0 and 1, respectively. This imbalance was visually evident in the plot, illustrating the relative frequency of each category. Understanding this distribution is crucial, as it guides our model-building process.
     
     ![churners_vs_non_churners](https://github.com/nganibaniathule/Portfolio/assets/129146143/fc8a5da6-4c71-4e65-aac6-58305c37415d)


- Geographical Impact:
   - Geographical analysis revealed intriguing findings. Notably, Spain exhibited the lowest churn rate compared to other countries, such as France and Germany. This observation prompts further investigation into the regional factors influencing customer retention.
     
     ![countries](https://github.com/nganibaniathule/Portfolio/assets/129146143/124dedf2-820f-45f6-abce-32243c198e4d)


- Gender Influence:
   - Gender analysis indicated that females tend to churn more than males. This observation aligns with the fact that females often possess multiple credit cards and 
       subsequently make choices on credit card usage. This insight highlights the importance of
       tailoring retention strategies based on demographic factors.
     
    ![gender](https://github.com/nganibaniathule/Portfolio/assets/129146143/65fcae11-ee78-412a-83ec-dcf6837cb3c3)


- Product Number Effect:
   - The analysis revealed an intriguing correlation: as the product number increases, the likelihood of customers leaving the bank diminishes. This finding underscores the significance of product offerings in customer retention strategies.
     
     ![products_numbers](https://github.com/nganibaniathule/Portfolio/assets/129146143/48840de8-4cc7-44c8-8672-4e6abef399b5)


- Credit Card Impact:
   - Customers with credit cards exhibited a higher rate of churning compared to those without credit cards. This insight raises questions about the relationship between credit card usage and churn, necessitating further exploration.
     
     ![credit_card](https://github.com/nganibaniathule/Portfolio/assets/129146143/f96b1a57-1d33-422a-a183-9a186a541b4d)


- Active Membership:
   - The data demonstrated that active members had a lower churn rate compared to non-active members. This distinction emphasizes the importance of customer engagement and involvement in bank services as a factor in retention.
     
     ![active_member](https://github.com/nganibaniathule/Portfolio/assets/129146143/a133e19a-55c2-44b7-b539-a9aee1e362ed)


Correlation Analysis:
   - Our calculations revealed that the variables in our dataset are not significantly correlated with each other. This finding suggests that the predictors operate independently and might contribute unique information to the predictive model.
     
     ![correlation_matrix](https://github.com/nganibaniathule/Portfolio/assets/129146143/afd574c3-e47b-442d-a2b1-8fc7f3271c89)


Our EDA provides a critical foundation for the subsequent stages of the project, guiding feature selection, model development, and retention strategy formulation. The insights gained from this initial exploration equip us with a deeper understanding of the dataset and empower data-driven decisions in our pursuit of effective customer churn prediction.

## Feature Engineering:

In the pursuit of creating an optimal predictive model for Bank Customer Churn, feature engineering played a pivotal role in refining our dataset. This section outlines the steps taken to enhance the data's suitability for predictive analysis.

- Column Removal:
   * To streamline our dataset and ensure that we focus on the most relevant features, we made the deliberate choice to remove the 'customer_id' column. This decision was informed by the understanding that the 'customer_id' attribute does not contribute any valuable information to our churn predictions.

- Balancing the Dataset:
   * Addressing class imbalance is a critical aspect of predictive modeling. We employed two prominent techniques, NearMiss and SMOTE (Synthetic Minority Over-sampling Technique), to address this challenge. These techniques aim to balance the distribution of the two classes (churners and non-churners) in the dataset. Our results demonstrated that SMOTE was the preferred choice, as it provided notably improved outcomes and a balanced dataset.

The feature engineering steps taken in this phase were instrumental in refining the dataset, ensuring class balance, and preparing the data for modeling. This strategic preparation forms the basis for the subsequent development and evaluation of our predictive model for Bank Customer Churn.
