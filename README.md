<!DOCTYPE html>

<html lang="en">

<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <link rel="stylesheet" href="style.css">
  
    <title>Llama 2 Meta's Large Language Model</title>
    <style>
        link rel="stylesheet" href="style.css">
        #about img {
            display: block;
            margin: 0 auto;
        }

    </style>
    
body {
    font-family: Arial, sans-serif;
    margin: 0;
    padding: 0;
    background-color: #ADD8E6; 
    color: #333; /* Text color */
}

/* Header styles */
header {
    background-color: #333; /* Dark blue header background */
    color: #fff; /* Header text color */
    text-align: center;
    padding: 1em 0;
}

/* Navigation styles */
nav {
    background-color: #333; /* Dark gray navigation background */
    color: #fff; /* Navigation text color */
    padding: 0.5em;
    text-align: center;
}

nav a {
    color: #fff; /* Link text color */
    text-decoration: none;
    margin: 0 10px;
}

/* Section styles */
section {
    margin: 2em;
    padding: 1em;
    background-color: #fff; /* White section background */
    border-radius: 5px;
    box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);
}

/* Footer styles */
footer {
    text-align: center;
    padding: 1em 0;
    background-color: #333; /* Dark gray footer background */
    color: #fff; /* Footer text color */
}

/* Link styles */
a {
    color: #007bff; /* Link color */
    text-decoration: none;
}

a:hover {
    text-decoration: underline;
}

</head>
<body>
    <header>
        <h1>Welcome to the Decision Trees algorithm</h1>
    </header>

    <nav>
        <!-- Add navigation links here -->
    </nav>

    <section id="about">
        <h2>Decision Trees</h2>
        <p>
            
Decision Trees are a fundamental machine learning algorithm used for solving classification and regression tasks.
In essence, a Decision Tree is a flowchart-like structure where internal nodes represent feature-based decisions, branches represent possible outcomes, and leaf nodes represent the final predicted class or value.
The primary objective of using Decision Trees is to divide the data into subsets based on different features, leading to distinct and homogeneous groups that can be easily classified or predicted.
These algorithms are widely used in various real-world applications, such as:
Customer Churn Prediction: Determining the likelihood of customers leaving a service or product.
Medical Diagnosis: Assisting medical professionals in diagnosing diseases based on patient symptoms.
Financial Risk Assessment: Assessing the risk associated with investments or loans for financial institutions.
Decision Trees offer great interpretability, as the decision-making process can be visually represented and understood, making them popular in decision support systems.
</p>

<p>
            Llama 2 is still under development, but it has the potential to be a powerful tool for a variety of applications. It could be used to build more realistic chatbots, create new forms of creative content, and even help us to better understand the world around us.
        </p>
    </section>

    <section id="applications">
        <h2>Potential Applications of Decision Trees</h2>
        <ul>
            <li>Classification Problems: Decision trees are often used to solve classification problems where the goal is to categorize input data into different classes. For example:

Email spam detection: Classify emails as spam or not spam based on various features like sender, subject, and content.
Medical diagnosis: Classify patients into different disease categories based on their symptoms and medical history.
Sentiment analysis: Determine the sentiment of a text (positive, negative, neutral) based on the words and phrases used. </li>
            <li>Regression Problems: Decision trees can also be used for regression tasks, where the goal is to predict a continuous numeric value. For instance:

House price prediction: Predict the price of a house based on features like location, size, and amenities.
Demand forecasting: Predict future sales or demand for a product based on historical sales data and other relevant factors.</li>
            <li>Feature Selection: Decision trees can help identify the most important features that contribute to the decision-making process. Features with high information gain are often more relevant for making accurate predictions.</li>
            <li>Anomaly Detection: Decision trees can be used to identify anomalies or outliers in a dataset by finding instances that deviate significantly from the expected patterns.</li>
        </ul>
         <div style="text-align: center;"> <!-- Centering the content -->
            <img src="llama2.png" alt="Llama 2 Image" width="300" height="auto">
        </div>
        <p>
         Decision trees are a popular machine learning algorithm used for both classification and regression tasks. They can be applied in a variety of domains and industries for various purposes.
            
        </p>
    </section>

    <section id="papers">
        <h2>Research Papers</h2>
        <ul>
            <li><a href="https://www.researchgate.net/publication/225237661_Decision_Trees">Paper 1 : Original paper by Developers</a></li>
            <li><a href="https://hunch.net/~coms-4771/quinlan.pdf">Paper 2:   Review paper on Fine tuned Large Language Models</a></li>
            <!-- Add more paper references as needed -->
        </ul>
    </section>

    <section id="videos">
        <h2>Videos</h2>
        <ul>
            <li><a href="https://www.youtube.com/watch?v=RVuy1ezN_qA">Video 1: Introduction to Decision Tree</a></li>
            <li><a href="https://www.youtube.com/watch?v=LFTeAnPcgUw">Video 2:  Decision Tree Basics</a></li>
            <!-- Add more video references as needed -->
        </ul>
    <section id="code-section">
    <h2>Python Code Example</h2>
    <p>Below is an example of Python code </p>
    <pre>
        <code>
# -- coding: utf-8 --
"""Llama2 finetuning on Instacart Dataset-Parikshit Sangar

# Importing necessary libraries
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Load the Iris dataset (a popular example dataset)
iris = load_iris()
X = iris.data
y = iris.target

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a Decision Tree classifier
clf = DecisionTreeClassifier(random_state=42)

# Train the classifier on the training data
clf.fit(X_train, y_train)

# Make predictions on the test data
y_pred = clf.predict(X_test)

# Calculate the accuracy of the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

        </code>
    </pre>
</section>

    <section id="presentation">
        <h2>Embedded Presentation</h2>
        <iframe src="https://drive.google.com/drive/folders/1H7xlE0aAD12TGBoL4LIssDrh1f48pTIE?usp=drive_link" width="800" height="600" frameborder="0" allowfullscreen="true" mozallowfullscreen="true" webkitallowfullscreen="true"></iframe>
    </section>

    <footer>
        <p>&copy; 2023 Your Name. All rights reserved.</p>
    </footer>
</body>
</html>
