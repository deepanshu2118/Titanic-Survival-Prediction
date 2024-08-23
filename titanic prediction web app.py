import numpy as np
import pickle
import streamlit as st

loaded_model = pickle.load(open('C:/Users/LENOVO/OneDrive/Desktop/some project with streamlit/titanic_project/titanic.pkl', 'rb'))

def titanic_prediction(input_data):
    input_data_as_numpy_array=np.asarray(input_data)
                                            
    input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

    prediction=loaded_model.predict(input_data_reshaped)
    print(prediction)

    if (prediction[0]==0):
        return 'The person in no alive'
    else:
        return 'The person is alive'
    
def main():
    st.title("titanic Classification:")

    Pclass=st.slider("Pclass:",min_value=1, max_value=3,step=1)
    Sex=st.slider("Sex:",min_value=0,max_value=1,step=1)
    Age=st.slider("Age:",min_value=1,max_value=100,step=1)
    Fare=st.slider('Fare:',min_value=0.0,max_value=100.0,step=0.1)
    Embarked=st.slider("Embarked:",min_value=0,max_value=2,step=1)
    SibSp=st.slider("SibSp:",min_value=0,max_value=5,step=1)
    Parch=st.slider("Parch:",min_value=0,max_value=5,step=1)


    titanisity=''

    if st.button('titanic test result'):
        titanisity= titanic_prediction([Pclass,Sex,Age,Fare,Embarked,SibSp,Parch])

    st.success(titanisity)



if __name__ == '__main__':
    main()