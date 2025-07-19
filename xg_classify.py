import pandas as pd
import sqlite3
import datetime
from xgboost import XGBClassifier as XGB
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.linear_model import LogisticRegression as LR

from sklearn.model_selection import train_test_split as TTS 
from sklearn.metrics import accuracy_score as AS
from sklearn.metrics import confusion_matrix as CM
from sklearn.metrics import classification_report as CR
from sklearn.datasets import load_iris as LI
from sklearn.preprocessing import StandardScaler as SS

#upload data to dataframe
df_fact = pd.read_csv('ad_click.csv')

#create SQL database and upload dataframe
connection = sqlite3.connect('sql_db.db')
df_fact.to_sql('ad_click', connection, if_exists='replace')

# Define the date range for filtering
start_date = '2015-01-01'
end_date = '2017-12-31'

# SQL query to filter data based on the date range and create quintiles
sql_1 = """
    With base_table as (

        select * from ad_click k
        where date(k.Timestamp) >= date('{sd}') and date(k.Timestamp) <= date('{ed}')
    )

    select *,
    case 
    when g.Hour_of_Day >= 0 and g.Hour_of_Day < 6 then 0 
    when g.Hour_of_Day >= 6 and g.Hour_of_Day < 12 then 1 
    when g.Hour_of_Day >= 12 and g.Hour_of_Day < 18 then 2 
    when g.Hour_of_Day >= 18 and g.Hour_of_Day < 24 then 3 
    else 4 
    end as Time_of_Day

    from 
    (
        select  *,
        NTILE(5) OVER (ORDER BY j.Area_Income) AS Area_Income_Quintile,
        NTILE(5) OVER (ORDER BY j.Daily_Internet_Usage) AS Daily_Internet_Usage_Quintile,
        NTILE(5) OVER (ORDER BY j.Daily_Time_Spent_on_Site) AS Daily_Time_Spent_on_Site_Quintile,
        NTILE(5) OVER (ORDER BY j.Age) AS Age_Quintile,
        CAST(strftime('%H', j.Timestamp) AS INTEGER) AS Hour_of_Day
        from base_table j
    ) g



    

""".format(sd = start_date, ed = end_date)
sql_output = connection.execute(sql_1)
df_sql_output = pd.DataFrame(sql_output.fetchall())

# Set the column names for the DataFrame
df_sql_output.columns = [
        'ROW',
        'Daily_Time_Spent_on_Site',
        'Age',
        'Area_Income',
        'Daily_Internet_Usage',
        'spend_segmentAd_Topic_Line',
        'City',
        'Male',
        'Country',
        'Timestamp',
        'Clicked_on_Ad',
        'Area_Income_Quintile',
        'Daily_Internet_Usage_Quintile',
        'Daily_Time_Spent_on_Site_Quintile',
        'Age_Quintile',
        'Hour_of_Day',
        'Time_of_Day'
    ]

# Prepare the data for modeling
y = df_sql_output['Clicked_on_Ad']

#Exclude the target variable and other non-feature columns
X = df_sql_output.drop(columns=[
        'ROW',
        'Daily_Time_Spent_on_Site',
        'Age',
        'Area_Income',
        'Daily_Internet_Usage',
        'spend_segmentAd_Topic_Line',
        'City',
        #'Male',
        'Country',
        'Timestamp',
        'Clicked_on_Ad',
        #'Area_Income_Quintile',
        #'Daily_Internet_Usage_Quintile',
        #'Daily_Time_Spent_on_Site_Quintile',
        #'Age_Quintile',
        'Hour_of_Day'
        #'Time_of_Day'
    
    ])

# Standardize the features
X_train, X_test, y_train, y_test = TTS(X, y, test_size=0.2, random_state=42)

model = XGB(learning_rate=0.1, n_estimators=100, max_depth=3, random_state=42)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

# Print feature importances
print("Feature Importances:")
for name, importance in zip(X.columns, model.feature_importances_):
    print(f"{name}: {importance}")

# Evaluate the model
accuracy = AS(y_test, y_pred)
conf_matrix = CM(y_test, y_pred)

print(f"Accuracy: {accuracy:.2f}")
print("Confusion Matrix:")
print(conf_matrix)

print('Done')


