# Bank Customer Churn Prediction Project
## Introduction

In an era of fierce competition in the banking industry, retaining customers has become a paramount concern for financial institutions. Customer churn is the process by which customers cease their association with a bank. It can have significant repercussions on a bank's profitability and overall performance. The ability to predict and mitigate customer churn is, therefore, a strategic imperative for any modern bank.

Bank customer churn prediction is a critical application of data science and machine learning techniques. It forecasts which customers are likely to leave a bank in the near future. This predictive modelling not only serves as a proactive measure to retain customers but also as an opportunity to enhance customer satisfaction and experience.

The goal of this project is to develop a robust predictive model that identifies at-risk customers by analyzing various features such as transaction history, demographics, account activity, and customer interactions. By understanding the underlying patterns and behaviours of customers who have previously churned, banks can implement targeted retention strategies, offer personalized incentives, and ultimately reduce the attrition rate.

This portfolio project delves into the realm of bank customer churn prediction. It showcases the application of data preprocessing, feature engineering, and machine learning algorithms to build a predictive model. Through this project, we explore the significance of data-driven decision-making in the financial sector and demonstrate the potential impact of predictive analytics on customer retention and revenue growth.

Join me on this journey to harness the power of data science to address one of the most pressing challenges faced by the banking industry: predicting and preventing customer churn.


## Problem Statement

"In the dynamic landscape of the banking industry, customer churn, or the loss of valued clients, has emerged as a pressing concern for financial institutions. As a proactive response to this challenge, our project aims to develop an effective predictive model for bank customer churn. The primary objective is to identify and predict potential churners among the bank's existing customer base, allowing the bank to take targeted actions to retain these customers.

The project entails the utilization of data science and machine learning techniques to create a predictive model that can accurately flag individuals at risk of churning. By analyzing a diverse set of customer-related data, including transaction history, demographics, account activity, and customer interactions, we seek to uncover underlying patterns, behaviours, and factors contributing to churn.

The key questions we aim to address are:
- Can we accurately predict which customers are likely to churn in the near future?
* What features and factors are significant predictors of churn?
+ How can this predictive model be leveraged to implement retention strategies and improve overall customer satisfaction?

Our ultimate goal is to equip banks with a predictive tool that aids in the identification of customers at risk of leaving. Thereby empowering them to deploy personalized engagement and retention initiatives. Through the development of this model, we aspire to not only reduce customer churn but also enhance customer relationships and financial performance in the banking sector."

## Data Collection

The dataset used in this project was sourced from Kaggle, a renowned platform for data science and machine learning resources. It is essential to note that the dataset was already in a well-prepared and clean state, with no missing values, making it readily usable for analysis and model development. The data was provided in the form of a CSV (Comma-Separated Values) file, a common and easily accessible data format in the Python programming environment.

This streamlined data collection process facilitated efficient data exploration, preprocessing, and the development of a robust predictive model. The CSV format's compatibility with Python allowed for seamless integration into the project, enabling straightforward data manipulation, visualization, and machine learning model building.

## Data Preprocessing

The dataset was exceptionally well-prepared, requiring minimal initial data cleaning. Notably, there were no missing or null values in the dataset, ensuring the integrity of the information used for analysis and modelling.

To harness the predictive power of non-numeric features and enable their inclusion in our models, we embarked on an encoding process. Non-numeric columns were converted into categorical variables, allowing us to integrate these essential aspects of the data into our analysis effectively.

However, the dataset presented a challenge in the form of numeric columns with varying ranges. This diversity in scale among numerical attributes can potentially hinder the performance of machine learning models. To address this issue, we applied the MinMaxScaler, a transformation technique provided by the Scikit-Learn library. This scaler normalized the numeric columns, ensuring that all features were on the same scale within a specified range. This preprocessing step was crucial in enhancing the model's ability to effectively interpret and weigh each feature, ultimately contributing to more accurate predictions.

The combination of encoding non-numeric columns and scaling numerical ones laid a solid foundation for our subsequent analysis and predictive modelling efforts.


## Exploratory Data Analysis (EDA)

During the exploratory phase of our analysis, we gained valuable insights into the dataset, shedding light on patterns and trends that form the foundation of our Bank Customer Churn Prediction project.

- Churn Distribution:
   - Our dataset comprises 7,963 non-churners (0) and 2,037 churners (1). This imbalance was visually evident in the plot, illustrating the relative frequency of each category. Understanding this distribution is crucial, as it guides our model-building process.
     
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
     
     ![correlation_matrix](https://github.com/nganibaniathule/Portfolio/assets/129146143/8e2d56c4-86d5-44b9-ae19-5a1474132051)



Our EDA provides a critical foundation for the subsequent stages of the project, guiding feature selection, model development, and retention strategy formulation. The insights gained from this initial exploration equip us with a deeper understanding of the dataset and empower data-driven decisions in our pursuit of effective customer churn prediction.

## Feature Engineering

In the pursuit of creating an optimal predictive model for bank customer churn, feature engineering played a pivotal role in refining our dataset. This section outlines the steps taken to enhance the data's suitability for predictive analysis.

- Column Removal:
   * To streamline our dataset and ensure that we focus on the most relevant features, we made the deliberate choice to remove the 'customer_id' column. This decision was informed by the understanding that the 'customer_id' attribute does not contribute any valuable information to our churn predictions.

- Balancing the Dataset:
   * Addressing class imbalance is a critical aspect of predictive modelling. We employed two prominent techniques, NearMiss and SMOTE (Synthetic Minority Over-sampling Technique), to address this challenge. These techniques aim to balance the distribution of the two classes (churners and non-churners) in the dataset. Our results demonstrated that SMOTE was the preferred choice, as it provided notably improved outcomes and a balanced dataset.

The feature engineering steps taken in this phase were instrumental in refining the dataset, ensuring class balance, and preparing the data for modelling. This strategic preparation forms the basis for the subsequent development and evaluation of our predictive model for Bank Customer Churn.


## Model Selection

In the quest to develop a robust bank customer churn prediction model, we embarked on the crucial process of selecting suitable machine learning algorithms. These algorithms are pivotal in our efforts to effectively identify potential churners and facilitate data-driven decision-making within the banking sector.

The models chosen for this project encompass a diverse range of machine learning techniques, each with unique strengths and capabilities:

1. **_Logistic Regression_**: A fundamental yet effective algorithm for binary classification, logistic regression provides interpretability and is well-suited for understanding the relationship between features and the likelihood of churn.

2. **_Naive Bayes Classifier_**: Leveraging probabilistic principles, this classifier is particularly useful when dealing with categorical features and is known for its simplicity and efficiency.

3. **_Decision Tree_**: Decision trees are interpretable models that segment data into hierarchical structures, offering insights into the factors influencing churn decisions.

4. **_Random Forest_**: As an ensemble technique, random forest aggregates the predictive power of multiple decision trees, providing robustness and reducing overfitting.

5. **_Support Vector Machine (SVM)_**: SVM excels in separating complex data into distinct classes. Its capacity for handling high-dimensional data makes it a valuable tool in the churn prediction context.

6. **_Linear Discriminant Analysis (LDA)_**: LDA is adept at dimensionality reduction and is particularly useful when working with correlated features.

7. **_Quadratic Discriminant Analysis (QDA)_**: Similar to LDA, QDA is employed for dimensionality reduction but allows for more flexibility in capturing non-linear relationships.

8. **_Gradient Boosting_**: Gradient boosting assembles a sequence of weak learners into a robust model, offering strong predictive performance and interpretability.

9. **_K-Nearest Neighbors (KNN)_**: KNN leverages the proximity of data points to make predictions and is especially useful in detecting localized patterns.

10. **_MLP Neural Network_**: As a deep learning approach, the Multilayer Perceptron (MLP) neural network excels in capturing complex, non-linear relationships within the data.

**Relevance to bank churn prediction:**
The selection of these diverse models allows us to thoroughly explore the complex landscape of Bank Customer Churn Prediction. By evaluating various algorithms, we gain insights into the performance, interpretability, and generalization capacity of each model. This extensive analysis equips us to make informed decisions about the best model for addressing the unique challenges posed by churn prediction in the banking industry.

Moreover, the chosen models cater to the multi-faceted nature of the dataset, addressing various types of features and relationships. This ensures that our analysis is comprehensive and sensitive to nuances in customer behavior, facilitating the development of accurate churn predictions and aiding in the formulation of tailored retention strategies.


## Model Training and Evaluation

The process of training and evaluating our predictive models for Bank Customer Churn Prediction is fundamental to the project's success. This section outlines the key steps and metrics employed to ensure the robustness and reliability of our models.

**Data Splitting:**
To facilitate effective model training and assessment, the dataset was thoughtfully divided into two distinct sets: a training dataset, which accounted for 80% of the data, and a testing dataset, which represented the remaining 20%. This split ratio is a common practice in machine learning. A higher proportion of data in the training set ensures that models learn sufficiently from the data, capturing the complexity of the dataset and its underlying patterns.

**Choice of Evaluation Metrics:**
In this project, the choice of evaluation metrics was a deliberate decision aimed at addressing the specific nature of the Bank Customer Churn Prediction problem.

1. **_Recall:_** Recall, also known as sensitivity or true positive rate, was selected as a crucial metric. Recall quantifies the ability of our models to correctly identify non-churners (true positives) out of all actual non-churners, minimizing the risk of failing to detect potential churners. In this context, true positives represent non-churners, and optimizing recall ensures that we accurately identify customers who are likely to stay with the bank.

2. **_Precision:_** Precision, or positive predictive value, gauges the ability of our models to correctly classify true positives (non-churners) out of all instances predicted as non-churners. It is pivotal in minimizing false positives and ensuring that our retention strategies are cost-effective and precisely targeted.

3. **_Accuracy:_** Although accuracy is a common evaluation metric, its relevance in this project stems from the balanced nature of the dataset. With a balanced dataset, models are less likely to be biased toward the majority class. Accuracy provides a comprehensive measure of the model's overall performance, taking into account both true positives (non-churners) and true negatives (churners).

**Interpretation of True Positives and True Negatives:**
It's worth noting that in the context of this project, the interpretation of true positives and true negatives is as follows:

- **True Positives (TP):** In the context of Bank Customer Churn Prediction, true positives represent non-churners, meaning customers correctly identified by our models as likely to stay with the bank.

- **True Negatives (TN):** True negatives represent churners, denoting customers correctly identified as likely to leave the bank. In this scenario, we aim to minimize the misclassification of churners as non-churners, as customer retention is the primary objective.

These chosen metrics and their interpretations are tailored to the specific challenges posed by Bank Customer Churn Prediction. By optimizing recall and precision while ensuring balanced performance via accuracy, we aim to strike a balance between retaining valuable customers and cost-effective retention strategies.


## Results and Insights
### Results
The evaluation of multiple machine learning models has yielded valuable results, each offering a unique perspective on the challenges of Bank Customer Churn Prediction. This section provides an overview of the performance metrics for the selected models and the actionable insights derived from these results.

**Model Performance Metrics:**

1. **_Random Forest:_**
   - Recall = 61%
   - Precision = 58%
   - Accuracy = 84%

2. **K-Nearest Neighbors (KNN):**
   - Recall = 63%
   - Precision = 38%
   - Accuracy = 73%

3. **Naive Bayes Classifier:**
   - Recall = 69%
   - Precision = 40%
   - Accuracy = 74%

4. **Gradient Boosting:**
   - Recall = 69%
   - Precision = 57%
   - Accuracy = 84%

5. **Logistic Regression:**
   - Recall = 71%
   - Precision = 39%
   - Accuracy = 72%

6. **Linear Discriminant Analysis:**
   - Recall = 71%
   - Precision = 39%
   - Accuracy = 73%

7. **Support Vector Machine (SVM):**
   - Recall = 71%
   - Precision = 46%
   - Accuracy = 78%

8. **MLP Neural Network:**
   - Recall = 71%
   - Precision = 47%
   - Accuracy = 78%


### Actionable Insights

1. **Model Selection:**
   - Among the evaluated models, several stand out in terms of recall, precision, and accuracy. Notably, Logistic Regression, Linear Discriminant Analysis, SVM, and MLP Neural Network all achieved a recall rate of 71%, suggesting their effectiveness in identifying non-churners. Considering precision and accuracy, Random Forest, Gradient Boosting, SVM, and MLP Neural Network demonstrated strong performance.

2. **Balancing Precision and Recall:**
   - The choice of the best model should be guided by the bank's specific priorities. Models with high recall, such as Logistic Regression, are effective at identifying non-churners. This is crucial for retaining valuable customers. On the other hand, models like Random Forest and Gradient Boosting offer a balance between recall, precision, and accuracy, making them suitable for optimizing both customer retention and cost-effective strategies.

3. **Ensemble Learning:**
   - Considering the promising results of ensemble models like Random Forest and Gradient Boosting, it may be prudent to explore ensemble techniques further. These models offer a blend of predictive power, interpretability, and robustness.

4. **Predictive Insights:**
   - The insights drawn from these models can guide the bank's strategies for customer retention. Understanding the drivers of churn and identifying customers at risk can inform tailored interventions, such as personalized offers, enhanced customer service, and improved engagement initiatives.

5. **Ongoing Monitoring:**
   - The performance of the selected model should be monitored over time as customer behavior evolves. Ongoing data collection and retraining of the model can help adapt retention strategies to changing customer dynamics.

In summary, the choice of the best model hinges on the bank's specific objectives and the trade-off between identifying potential churners and minimizing false positives. Whether prioritizing recall, precision, or a balanced approach, the insights gained from these results provide a valuable foundation for informed decision-making and effective customer churn prediction strategies.


## Model Deployment

Deploying the best predictive model in a real-world banking scenario involves integrating it into the bank's operational framework for effective decision-making. The selected model, based on its performance and relevance to the bank's objectives, can be implemented as part of an automated system or tool used by the bank's customer relationship management team. This tool can continually analyze customer data to identify potential churners, enabling proactive, data-driven interventions. For instance, when a customer is flagged as at risk of churning, the system can trigger personalized retention strategies, such as offering tailored incentives, optimizing product recommendations, or enhancing customer support. Real-time monitoring and periodic model updates ensure that the bank's customer retention efforts remain agile and responsive to evolving customer behaviours. This deployment empowers the bank to take proactive measures to retain valuable customers, thereby contributing to improved customer satisfaction and long-term profitability.


## Conclusion

In the dynamic landscape of the banking industry, customer retention has emerged as a critical determinant of a bank's success and profitability. This project has revolved around the development of a predictive model for Bank Customer Churn, a task that holds profound implications for the industry.

The key findings of this endeavour are as follows:

- We conducted a comprehensive analysis of customer data, harnessed machine learning models, and carefully selected evaluation metrics tailored to the specific challenges posed by the churn prediction problem.

- Among the evaluated models, various options demonstrated commendable performance. Notably, models like Logistic Regression, Linear Discriminant Analysis, SVM, and MLP Neural Network displayed a high recall rate, effectively identifying non-churners.

- Ensemble models, such as Random Forest and Gradient Boosting, showcased the potential to balance recall, precision, and accuracy, making them suitable for comprehensive customer retention strategies.

The impact of this project on customer churn prediction is significant. It equips banks with a powerful tool for understanding and anticipating customer behavior. By harnessing the insights provided by our predictive model, banks can proactively retain valuable customers and enhance overall customer satisfaction. The project's implications for the banking industry are profound, as it enables financial institutions to formulate data-driven, personalized retention strategies, reduce customer attrition, and sustain long-term growth and profitability.

In the face of evolving customer preferences and a highly competitive environment, predictive analytics and data-driven insights have become indispensable tools for banks to stay ahead. The continuous monitoring and adaptation of predictive models, in conjunction with well-informed retention strategies, are pivotal for thriving in the ever-evolving landscape of the banking sector. This project has illuminated a path toward harnessing the power of data science for customer churn prediction, offering both valuable insights and the means to translate these insights into real-world impact.


## Future Work

While this project has provided valuable insights and predictive capabilities for Bank Customer Churn Prediction, there are several avenues for future work and enhancements to further advance the field and its applications in the banking industry:

1. **_Real-Time Monitoring:_** Implementing real-time data streaming and monitoring to enable instant identification of churn risk. This would allow banks to respond to customer behavior changes promptly.

2. **_Automated Retention Strategies:_** Developing an automated system that not only identifies potential churners but also triggers and monitors personalized retention strategies in real time, optimizing customer engagement.

3. **_Customer Segmentation:_** Extending the analysis to incorporate more advanced customer segmentation techniques to tailor retention strategies to specific customer profiles and behaviors.

4. **_Natural Language Processing (NLP):_** Incorporating NLP techniques to analyze unstructured data, such as customer feedback, reviews, and social media interactions, to gain additional insights into customer sentiment and preferences.

5. **_Deep Learning:_** Exploring advanced deep learning models for improved feature extraction and predictive performance. Techniques like recurrent neural networks (RNNs) and long short-term memory networks (LSTMs) may offer enhanced predictive capabilities.

6. **_Evaluating External Data Sources:_** Investigating the inclusion of external data sources, such as economic indicators, regional data, or customer social profiles, to gain a more comprehensive understanding of customer churn drivers.

7. **_Ethical Considerations:_** Delving into ethical considerations surrounding customer data usage, privacy, and fairness, ensuring that the project adheres to regulatory and ethical guidelines.

8. **_Cost-Benefit Analysis:_** Conducting a cost-benefit analysis to evaluate the financial impact of various retention strategies and help banks allocate resources more efficiently.

9. **_Customer Feedback Loop:_** Establishing a feedback loop to incorporate customer feedback and outcomes from retention strategies into the model's training and refinement process.

10. **_Global Expansion:_** Adapting the model and strategies for global or multi-market application, considering cultural and regional differences in customer behavior.

These avenues for future work not only expand the scope of customer churn prediction but also contribute to the continued evolution of data-driven decision-making in the banking industry. As customer preferences and market dynamics continue to evolve, staying at the forefront of predictive analytics and customer retention strategies remains a strategic imperative for financial institutions.
