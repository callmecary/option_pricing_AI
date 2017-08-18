import pandas as pd
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

model = load_model('trained_model.h5')
training_data_df = pd.read_csv('keras_train.csv')
scaler = MinMaxScaler(feature_range=(0,1))
scaled_training = scaler.fit_transform(training_data_df)
new_option_df = pd.read_csv("keras_evaluation.csv")
new_option_scaled = scaler.transform(new_option_df)
scaled_evaluation_df = pd.DataFrame(new_option_scaled,columns=new_option_df.columns.values)

X = scaled_evaluation_df.drop('premium',axis=1).values

prediction = model.predict(X)
prediction = prediction - scaler.min_[6]
prediction = prediction / scaler.scale_[6] 

print("Predicted premium is  ${}".format(prediction))