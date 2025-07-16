import streamlit as st
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, mean_squared_error, recall_score, precision_score, roc_auc_score, classification_report

st.set_page_config(page_title="ML Agent App", page_icon="🤖")
st.title('Classification Machine Learning App 🕵️‍♀️')

st.sidebar.title('Navigation')
options = ['Instruction','Upload Data','Select Features','Model Traning','Results']
selection = st.sidebar.radio('Go to', options)

def trainmodel():
    df = st.session_state['df']
    target = st.session_state['target']

    cat_col = df.drop(columns=target).select_dtypes(include=['category','object']).columns.tolist()
    num_col = [col for col in df.columns if col not in cat_col+[target]]
    df_getdum = pd.get_dummies(df[cat_col]) if cat_col else pd.DataFrame()  
    df_num = df[num_col]
    df_dumm = pd.concat([df_getdum,df_num],axis=1)
    
    le = LabelEncoder()
    X = df_dumm
    y = le.fit_transform(df[target])
    
    test_size = st.slider('Test size (%)',10,50,20)
    random_state = st.number_input('Random State', value=1994, step=1)
    
    models = {
    'Logistic Regression': LogisticRegression(multi_class='auto', solver='lbfgs', max_iter=1000),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'Support Vector Machine': SVC(kernel='rbf', probability=True),
    'K-Nearest Neighbors': KNeighborsClassifier(n_neighbors=5)
}
    
    selected_models = st.multiselect('Select models to train',list(models.keys()), default=list(models.keys()))
    
    if st.button('Train models'):
        if not selected_models:
            st.error('Please select atlest one model')
        else:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size/100, random_state=random_state)
            
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)
            
            st.session_state['X_train'] = X_train
            st.session_state['X_test'] = X_test
            st.session_state['y_train'] = y_train
            st.session_state['y_test'] = y_test
            st.session_state['scaler'] = scaler
            
            st.success('Data successfully split!')
            
            trained_models = {}
            for model_name in selected_models:
                model = models[model_name]
                model.fit(X_train, y_train)
                trained_models[model_name] = model
                
            st.session_state['trained_models'] = trained_models
            st.success('All models have been trained')

def visual(model_name, model, X_test, y_test):
    st.subheader(f'{model_name}')
    y_pred = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    df_report = pd.DataFrame(report).transpose()
    
    st.write(f'**accuracy:** {accuracy:.2f}')
    st.write(f'**recall:** {recall:.2f}')
    st.write(f'**precision:** {precision:.2f}')
    st.write(f'**mse:** {mse:.2f}')
    st.write(f'**auc:** {auc:.2f}')
    st.write('**classification report: **')
    st.dataframe(df_report.style.format("{:.2f}"))

def main():
    if selection == 'Instruction':
        st.write("""**Hi, welcome to my ML app!** \n
                Please follow these steps below to use this: \n
                - Step 1: Choose Upload Data tab to upload your dataset \n
                - Step 2: Define your features and target variable to train in models \n
                - Step 3: Training model with Model Training tab (You can choose multiple models) \n
                - Step 4: Checkout the results \n \n
                Good luck & have fun 😼
                """)

    if selection == 'Upload Data':
        st.header('Upload your dataset')
        uploaded_file = st.file_uploader('Upload a csv file', type=['csv'])
        
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            st.write('### Data Preview')
            st.dataframe(df.head())
            
            st.session_state['df'] = df
            
    if selection == 'Select Features':
        st.header('Select features and target variable')
        
        if 'df' not in st.session_state:
            st.warning('Please upload a dataset')
        else:
            df = st.session_state['df']
            all_columns = df.columns.tolist()
            
            target = st.selectbox('Select target variable', all_columns)
            features = st.multiselect('Select features', [col for col in all_columns if col != target], default=[col for col in all_columns if col != target])
            
            if st.button('Confirm selection'):
                if not features:
                    st.error('Please select atleast one feature')
                else:
                    st.session_state['features'] = features
                    st.session_state['target'] = target
                    st.success('Feature and target variable selected successfully')
                    
    if selection == 'Model Traning':
        st.header('Train ML models')
        
        if 'df' not in st.session_state or 'features' not in st.session_state or 'target' not in st.session_state:
            st.warning('Please upload and select features first')
        else:
            trainmodel()
            
    if selection == 'Results':
        st.header('Model performance results')
        
        if 'trained_models' not in st.session_state:
            st.warning('Please train model first')
        else:
            trained_models = st.session_state['trained_models']
            X_test = st.session_state['X_test']
            y_test = st.session_state['y_test']
        
            for model_name, model in trained_models.items():    
                visual(model_name, model, X_test, y_test)
                
if __name__ == '__main__':
    main()