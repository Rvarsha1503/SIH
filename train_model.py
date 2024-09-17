import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score

#Load the dataset
data = pd.read_csv('new_data.csv')  

# List of features
time_slots = ['12:00 AM', '1:00 AM', '2:00 AM', '3:00 AM', '4:00 AM', '5:00 AM', '6:00 AM', '7:00 AM', '8:00 AM', 
              '9:00 AM', '10:00 AM', '11:00 AM', '12:00 PM', '1:00 PM', '2:00 PM', '3:00 PM', '4:00 PM',
              '5:00 PM','6:00 PM', '7:00 PM', '8:00 PM', '9:00 PM', '10:00 PM', '11:00 PM']

#Fitting into the timeslots
le_features = LabelEncoder()
le_features.fit(time_slots)  

#Changing in the input columns
data['Previous Time Slot 1'] = le_features.transform(data['Previous Time Slot 1'])
data['Previous Time Slot 2'] = le_features.transform(data['Previous Time Slot 2'])
data['Previous Time Slot 3'] = le_features.transform(data['Previous Time Slot 3'])

#Target prediction column
le_target = LabelEncoder()
le_target.fit(data['Predicted Time Slot'])  
data['Predicted Time Slot'] = le_target.transform(data['Predicted Time Slot'])

#Fillting data as features in x and prediction column in y
X = data[['Previous Time Slot 1', 'Previous Time Slot 2', 'Previous Time Slot 3']]  # Features
y = data['Predicted Time Slot']  # Target

#Split into training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Actual training of the model using xgboost model
model = XGBClassifier()
model.fit(X_train, y_train)

#Predicting
y_pred = model.predict(X_test)

#checking accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%")

#saving model
import joblib
joblib.dump(model, 'time_slot_predictor_model.joblib')

#predicting with custom manual input
new_data = pd.DataFrame({
    'Previous Time Slot 1': ['5:00 AM'], 
    'Previous Time Slot 2': ['5:00 AM'],
    'Previous Time Slot 3': ['8:00 PM']
})

#transforming into features
new_data['Previous Time Slot 1'] = le_features.transform(new_data['Previous Time Slot 1'])
new_data['Previous Time Slot 2'] = le_features.transform(new_data['Previous Time Slot 2'])
new_data['Previous Time Slot 3'] = le_features.transform(new_data['Previous Time Slot 3'])

#prediction
predictions = model.predict(new_data)

#transforming back into original time slot format for eeasy understanding
predicted_time_slot = le_target.inverse_transform(predictions)

#printing the results on the console
print(f"Predicted Time Slot: {predicted_time_slot[0]}")
