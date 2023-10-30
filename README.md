# Customer Data Cleanup and Analysis Challenge

## Background:
A company has provided you with a dataset containing customer information. The dataset is messy and contains inconsistencies and missing values. The company is looking for a junior data analyst to clean up the data, create a suitable data model, store the cleaned data, and perform basic analytics to derive insights.

## Challenge Description:

### Step 1: Data Cleansing
The dataset provided contains the following columns:
- CustomerID (unique identifier for each customer)
- FirstName
- LastName
- Email
- PhoneNumber
- Age
- Gender
- Address
- PurchaseAmount

Your task is to perform the following data cleansing operations:

- Remove duplicate records based on CustomerID.
- Remove rows where any essential column (CustomerID, FirstName, LastName, Email, PhoneNumber) is missing.
- Clean the Email column to ensure all emails are in a valid format.
- Handle missing and outlier values in the Age and PurchaseAmount columns appropriately (e.g., replace missing ages with the median, remove outliers in PurchaseAmount).
- Standardize the Gender column to have consistent values (e.g., 'Male', 'Female', 'Other').

### Step 2: Data Modeling
Create a suitable data model to represent the cleaned customer data. Define the schema and relationships between different entities.

### Step 3: Data Storage
Store the cleaned and modeled data into a SQL database of your choice. Design an appropriate database schema to store customer information efficiently.

### Step 4: Data Analytics
Perform the following analytics tasks on the cleaned data:

- Calculate the average age of customers.
- Find the top 5 customers with the highest purchase amounts.
- Analyze the distribution of genders among the customers.
- Calculate the total purchase amount for each gender category.

## Deliverables:

- Python or SQL code for data cleansing, modeling, storage, and analytics.
- Explanation of the data model and schema design.
- Results of the analytics tasks.
- A brief report summarizing your findings and any insights derived from the analysis.

## Evaluation Criteria:

- Correctness and completeness of the data cleansing process.
- Appropriateness of the data model and database schema design.
- Proper storage of cleaned data in the SQL database.
- Accuracy and relevance of the analytics results.
- Quality and clarity of the code and explanations provided.

**Note:**
Feel free to use any programming languages, libraries, or tools you are comfortable with. The challenge is designed to assess your data cleansing, modeling, storage, and basic analytics skills.
