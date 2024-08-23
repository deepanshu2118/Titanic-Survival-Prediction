import pandas as pd 
import numpy as np
import pickle
loaded_model = pickle.load(open('C:/Users/LENOVO/OneDrive/Desktop/some project with streamlit/titanic_project/titanic.pkl', 'rb'))

input_data=(1,0,35,53.1000,2,1,0)

input_data_as_numpy_array=np.asarray(input_data)
                                            
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

prediction=loaded_model.predict(input_data_reshaped)
print(prediction)

if (prediction[0]==0):
    print('The person in no alive')
else:
    print('The person is alive')