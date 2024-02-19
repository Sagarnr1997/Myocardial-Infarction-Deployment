# -*- coding: utf-8 -*-
"""
Created on Sat May 13 01:25:07 2023

@author: Sagar N.R
"""
import streamlit as st
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.impute import KNNImputer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler
import os
import base64


@st.cache_data
def get_img_as_base64(file):
    with open(file, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()


with st.container():
    tabs =["About","User inputs","Predicting Test_dataset"]
    active_tab =st.radio("Select a tab",tabs)
    
    if active_tab == "About":
        st.markdown("""<style> body{background-color:#000000;color:#FFFFFF;}<style>""",unsafe_allow_html=True)
        html_temp = """
         <div style ="background-color:yellow;padding:10px">
         <h1 style ="color:black;text-align:center;"> About Myocardial infarction complications</h1>
         </div>
         """
        st.markdown(html_temp, unsafe_allow_html = True)
        image_file="http://upload.wikimedia.org/wikipedia/commons/f/fb/Blausen_0463_HeartAttack.png"
        st.image(image_file,caption="Heart attack",use_column_width=True)
        
        st.write('Myocardial infarction (MI) refers to tissue death (infarction) of the heart muscle (myocardium) caused by ischaemia, the lack of oxygen delivery to myocardial tissue. It is a type of acute coronary syndrome, which describes a sudden or short-term change in symptoms related to blood flow to the heart.[22] Unlike the other type of acute coronary syndrome, unstable angina, a myocardial infarction occurs when there is cell death, which can be estimated by measuring by a blood test for biomarkers (the cardiac protein troponin).[23] When there is evidence of an MI, it may be classified as an ST elevation myocardial infarction (STEMI) or Non-ST elevation myocardial infarction (NSTEMI) based on the results of an ECG.[24]The phrase "heart attack" is often used non-specifically to refer to myocardial infarction. An MI is different from—but can cause—cardiac arrest, where the heart is not contracting at all or so poorly that all vital organs cease to function, thus might lead to death.[25] It is also distinct from heart failure, in which the pumping action of the heart is impaired. However, an MI may lead to heart failure')
        html_temp = """
         <div style ="background-color:yellow;padding:10px">
         <h1 style ="color:black;text-align:center;"> Signs and symptoms</h1>
         </div>
         """
        image_file="http://upload.wikimedia.org/wikipedia/commons/2/24/AMI_pain_front.png"
        st.image(image_file,caption ='Pain',use_column_width=True)
        st.markdown(html_temp, unsafe_allow_html = True)
        st.write('Pain Chest pain is one of the most common symptoms of acute myocardial infarction and is often described as a sensation of tightness, pressure, or squeezing. Pain radiates most often to the left arm, but may also radiate to the lower jaw, neck, right arm, back, and upper abdomen.[28][29] The pain most suggestive of an acute MI, with the highest likelihood ratio, is pain radiating to the right arm and shoulder.[30][29] Similarly, chest pain similar to a previous heart attack is also suggestive.[31] The pain associated with MI is usually diffuse, does not change with position, and lasts for more than 20 minutes.[24] It might be described as pressure, tightness, knifelike, tearing, burning sensation (all these are also manifested during other diseases). It could be felt as an unexplained anxiety, and pain might be absent altogether.[29] Levine sign, in which a person localizes the chest pain by clenching one or both fists over their sternum, has classically been thought to be predictive of cardiac chest pain, although a prospective observational study showed it had a poor positive predictive value.[32] Typically, chest pain because of ischemia, be it unstable angina or myocardial infarction, lessens with the use of nitroglycerin, but nitroglycerin may also relieve chest pain arising from non-cardiac causes.[33] Chest pain may be accompanied by sweating, nausea or vomiting, and fainting,[24][30] and these symptoms may also occur without any pain at all.[28] In women, the most common symptoms of myocardial infarction include shortness of breath, weakness, and fatigue.[34] Women are more likely to have unusual or unexplained tiredness and nausea or vomiting as symptoms.[35] Women having heart attacks are more likely to have palpitations, back pain, labored breath, vomiting, and left arm pain than men, although the studies showing these differences had high variability.[36] Women are less likely to report chest pain during a heart attack and more likely to report nausea, jaw pain, neck pain, cough, and fatigue, although these findings are inconsistent across studies.[37] Women with heart attacks also had more indigestion, dizziness, loss of appetite, and loss of consciousness.[38] Shortness of breath is a common, and sometimes the only symptom, occurring when damage to the heart limits the output of the left ventricle, with breathlessness arising either from low oxygen in the blood, or pulmonary edema')
        html_temp = """
         <div style ="background-color:yellow;padding:10px">
         <h1 style ="color:black;text-align:center;"> Risk factors</h1>
         </div>
         """
        st.markdown(html_temp, unsafe_allow_html = True) 
        st.write('The most prominent risk factors for myocardial infarction are older age, actively smoking, high blood pressure, diabetes mellitus, and total cholesterol and high-density lipoprotein levels.[44] Many risk factors of myocardial infarction are shared with coronary artery disease, the primary cause of myocardial infarction,[16] with other risk factors including male sex, low levels of physical activity, a past family history, obesity, and alcohol use.[16] Risk factors for myocardial disease are often included in risk factor stratification scores, such as the Framingham Risk Score.[19] At any given age, men are more at risk than women for the development of cardiovascular disease.[45] High levels of blood cholesterol is a known risk factor, particularly high low-density lipoprotein, low high-density lipoprotein, and high triglycerides.[46] Many risk factors for myocardial infarction are potentially modifiable, with the most important being tobacco smoking (including secondhand smoke).[16] Smoking appears to be the cause of about 36% and obesity the cause of 20% of coronary artery disease.[47] Lack of physical activity has been linked to 7–12% of cases.[47][48] Less common causes include stress-related causes such as job stress, which accounts for about 3% of cases,[47] and chronic high stress levels.')
        html_temp = """
         <div style ="background-color:yellow;padding:10px">
         <h1 style ="color:black;text-align:center;">Business Objective:</h1>
         </div>
         """
        st.markdown(html_temp, unsafe_allow_html = True)
        st.write('Myocardial Infarction(Commonly known as “Heart attack”)  is one of the most challenging problems of modern medicine. Acute myocardial infarction is associated with high mortality in the first year after it. The incidence of MI remains high in all countries. This is especially true for the urban population of highly developed countries, which are exposed to chronic stress factors, irregular and not always balanced nutrition.The Given dataset consists of 124 variables with 1700 records of patients. Classify the Lethal outcome (cause) (LET_IS)(Y variable) by using the given dataset.')
        
    
    
    elif active_tab == "User inputs":
        page_bg_img = f"""
        <style>
        [data-testid="stAppViewContainer"] > .main {{
        background-image: url("https://cdna.artstation.com/p/assets/images/images/010/026/002/original/raveen-rajadorai-heart-beatingcycle.gif?1522160346");
        background-size: 140%;
        background-position: top left;
        background-repeat: repeat;
        background-attachment: local;
        }}
        [data-testid="stSidebar"] > div:first-child {{

        background-position: center; 
        background-no-repeat: no-repeat;
        background-attachment: fixed;
        }}
        [data-testid="stHeader"] {{
        background: rgba(0,0,0,0);
        }}
        [data-testid="stToolbar"] {{
        right: 2rem;
        }}
        </style>
        """
        html_temp = """
         <div style ="background-color:transperent;padding:13px">
         <h1 style ="color:black;text-align:center;">Myocardial infarction complications</h1>
         </div>
         """
        st.markdown('##')
        st.markdown(page_bg_img,unsafe_allow_html=True)
        st.markdown(html_temp, unsafe_allow_html = True)

    
        st.sidebar.header('User Input Parameters')
        
        def user_input_features():
            AGE  = st.sidebar.selectbox("Age of patient",(range(20,95,1)))
            st.sidebar.header('Select Gender 0 for female and 1 for male')
            SEX  = st.sidebar.selectbox("Gender",(0,1))
            STENOK_AN = st.sidebar.selectbox('Exertional angina pectoris',(0,1,2,3,4,5,6))
            ZSN_A = st.sidebar.selectbox('Presence of chronic HF',(0,1,2,3,4))
            K_SH_POST = st.sidebar.selectbox('Cardiogenic shock',(0,1))
            ant_im = st.sidebar.selectbox('Presence of an anterior myocardial infarction',(0,1,2,3,4))
            ritm_ecg_p_01 = st.sidebar.selectbox("ECG rhythm: sinus",(0,1))
            n_r_ecg_p_04 = st.sidebar.selectbox("Frequent premature ventricular contractions on ECG",(0,1))
            n_p_ecg_p_12 = st.sidebar.selectbox("Complete RBBB on ECG",(0,1))
            GIPO_K = st.sidebar.selectbox("Hypokalemia ( < 4 mmol/L)",(0,1))
            TIME_B_S = st.sidebar.selectbox("Time elapsed the attack of CHD",(1,2,3,4,5,6,7,8,9))
            R_AB_3_n = st.sidebar.selectbox("Relapse of the pain in the third day",(0,1,2,3))
            NITR_S = st.sidebar.selectbox("Use of liquid nitrates in the ICU",(0,1))
            NA_R_1_n = st.sidebar.selectbox("Use of opioid drugs in the ICU in the 1st day",(0,1,2,3,4))
            NOT_NA_2_n= st.sidebar.selectbox("Use of NSAIDs in the ICU in the 2nd day",(0,1,2,3))
            RAZRIV   = st.sidebar.selectbox("Myocardial rupture",(0,1))
            ZSN   = st.sidebar.selectbox("Chronic heart failure",(0,1))
            ROE   = st.sidebar.selectbox("Erythrocyte sedimentation rate",(range(0,150,1)))
            L_BLOOD   = st.sidebar.selectbox("White blood cell count (b/ltr)",(np.arange(2,30,0.1)))
            D_AD_ORIT   = st.sidebar.selectbox("Diastolic bp according to ICU (mmHg)",(range(0,190,1)))
           
            
            
            data = {
                    "Age of patient":AGE,
                    "Gender":SEX,
                    'STENOK_AN':STENOK_AN,
                    'ZSN_A':ZSN_A,
                    'K_SH_POST':K_SH_POST,
                    'ant_im':ant_im,
                    'ritm_ecg_p_01':ritm_ecg_p_01,
                    'n_r_ecg_p_04':n_r_ecg_p_04,
                    'n_p_ecg_p_12':n_p_ecg_p_12,
                    'GIPO_K':GIPO_K,
                    'TIME_B_S':TIME_B_S,
                    'R_AB_3_n':R_AB_3_n,
                    'NITR_S':NITR_S,
                    'NA_R_1_n':NA_R_1_n,
                    'NOT_NA_2_n':NOT_NA_2_n,
                    'RAZRIV':RAZRIV,
                    'ZSN':ZSN,
                    'ROE':ROE,
                    'L_BLOOD':L_BLOOD,
                    'D_AD_ORIT':D_AD_ORIT
                    }
            features = pd.DataFrame(data,index = [0])
            return features 
            
        df_1 = user_input_features()
        st.subheader('User Input parameters')
        st.write(df_1)
        
        
        df_new = pd.read_csv("Myocardial infarction complications.csv")
        df=df_new[['AGE','SEX','STENOK_AN','ZSN_A','K_SH_POST','ant_im','ritm_ecg_p_01','n_r_ecg_p_04','n_p_ecg_p_12','GIPO_K','TIME_B_S','R_AB_3_n','NITR_S','NA_R_1_n','NOT_NA_2_n','RAZRIV','ZSN','ROE','L_BLOOD','D_AD_ORIT','LET_IS']]
        df['AGE']=df.AGE.fillna(round(df.AGE.mean()))
        df['K_SH_POST']=df.K_SH_POST.fillna(round(df.K_SH_POST.mean()))
        df['NITR_S']=df.NITR_S.fillna(round(df.NITR_S.mean()))
        df['NA_R_1_n']=df.NA_R_1_n.fillna(round(df.NA_R_1_n.mean()))
        df['ritm_ecg_p_01']=df['ritm_ecg_p_01'].fillna(round(df.ritm_ecg_p_01.mean()))
        
        # KNN Imputation technique
        X = df.drop('LET_IS', axis=1)
        Y = df['LET_IS'] 
        imputer = KNNImputer(n_neighbors=5)
        col=X.columns
        df[col] = imputer.fit_transform(df[col])
        
        
        # Balancing & Model building
        # Train-Test-Split
        x = df.drop('LET_IS', axis=1)
        y = df['LET_IS']
        x=MinMaxScaler().fit_transform(df.iloc[:,:-1])
        x=pd.DataFrame(x,columns=df.columns.drop('LET_IS'))
        
        from imblearn.over_sampling import SMOTE
        
        
        # Apply SMOTE oversampling
        
        #train test split for the selected features
        X_train, X_test, y_train, y_test =train_test_split(x,y,test_size=0.3,random_state=42)
        
        
        model = LogisticRegression()
        classifier = OneVsRestClassifier(model)
        classifier.fit(X_train,y_train)
        
        prediction = classifier.predict(X_test)
        prediction_proba = classifier.predict_proba(X_test)
        
        
        st.write(f"<h2 style ='color:cyan;text-align:left;'>Accuracy of the Model is {np.round(accuracy_score(y_test, classifier.predict(X_test))*100,2)} %</h2>",unsafe_allow_html = True)
        
        
        if st.button("Predict"):   
            prediction = classifier.predict(df_1)
            st.write(f"<div style ='background-color:transparent;padding:2px'><h2 style ='color:cyan;text-align:left;'>Predicted Result of the Model is {prediction} class </h2> </div>",unsafe_allow_html = True)
            
            if prediction==0:
                st.write(f"<h2 style ='color:cyan;text-align:left;'> classified as alive </h2>",unsafe_allow_html = True)
            elif prediction==1:
                st.write(f"<h2 style ='color:cyan;text-align:left;'> classified as cardiogenic shock </h2>",unsafe_allow_html = True)
            elif prediction==2:
                st.write(f"<h2 style ='color:cyan;text-align:left;'> classified as pulmonary edema </h2>",unsafe_allow_html = True)
            elif prediction==3:
                st.write(f"<h2 style ='color:cyan;text-align:left;'> classified as myocardial rupture    </h2>",unsafe_allow_html = True)
            elif prediction==4:
                st.write(f"<h2 style ='color:cyan;text-align:left;'> classified as progress of congestive heart failure </h2>",unsafe_allow_html = True)
            elif prediction==5:
                st.write(f"<h2 style ='color:cyan;text-align:left;'> classified as thromboembolism  </h2>",unsafe_allow_html = True)
            elif prediction==6:
                st.write(f"<h2 style ='color:cyan;text-align:left;'> classified as asystole </h2>",unsafe_allow_html = True)
            else:
                st.write(f"<h2 style ='color:cyan;text-align:left;'> classified as ventricular fibrillation </h2>",unsafe_allow_html = True)
    
    elif active_tab == "Predicting Test_dataset":
        page_bg_img = f"""
        <style>
        [data-testid="stAppViewContainer"] > .main {{
        background-image: url("https://th.bing.com/th/id/R.bf0a8eb9f034bd8d98660331b893f33e?rik=kuQf7eyhzF4RZg&riu=http%3a%2f%2fwww.animated-gifs.fr%2fcategory_love%2fhearts-beating%2f10700882.gif&ehk=a%2fAFyxfL8p6%2fK597fP%2fgbevOH1tm94ykLWA3bECMBIA%3d&risl=&pid=ImgRaw&r=0");
        background-size: 100%;
        background-position: top left;
        background-repeat: repeat;
        background-attachment: local;
        }}
        [data-testid="stSidebar"] > div:first-child {{

        background-position: center; 
        background-no-repeat: no-repeat;
        background-attachment: fixed;
        }}
        [data-testid="stHeader"] {{
        background: rgba(0,0,0,0);
        }}
        [data-testid="stToolbar"] {{
        right: 2rem;
        }}
        </style>
        """
        html_temp = """
         <div style ="background-color:transperent;padding:13px">
         <h1 style ="color:black;text-align:center;">Myocardial infarction complications</h1>
         </div>
         """
        st.markdown('##')
        st.markdown(page_bg_img,unsafe_allow_html=True)
        st.markdown(html_temp, unsafe_allow_html = True)

        from imblearn.over_sampling import SMOTE
        
        df_new = pd.read_csv("Myocardial infarction complications.csv")
        df_new['AGE']=df_new.AGE.fillna(round(df_new.AGE.mean()))
        df_new['K_SH_POST']=df_new.K_SH_POST.fillna(round(df_new.K_SH_POST.mean()))
        df_new['NITR_S']=df_new.NITR_S.fillna(round(df_new.NITR_S.mean()))
        df_new['NA_R_1_n']=df_new.NA_R_1_n.fillna(round(df_new.NA_R_1_n.mean()))
        df_new['ritm_ecg_p_01']=df_new['ritm_ecg_p_01'].fillna(round(df_new.ritm_ecg_p_01.mean()))
        
        # KNN Imputation technique
        p = df_new.drop('LET_IS', axis=1)
        q = df_new['LET_IS'] 
        imputer = KNNImputer(n_neighbors=5)
        p = imputer.fit_transform(p)
        
        # Apply SMOTE oversampling
        smote = SMOTE(random_state=42)
        X_resampled, y_resampled = smote.fit_resample(p, q)
        
        from sklearn.ensemble import RandomForestClassifier
        
        final_model = RandomForestClassifier(n_estimators=100,max_depth=20,min_samples_split=2).fit(X_resampled, y_resampled)
        
        
        def load_data(file):
           data= pd.read_csv(file)
           return data
            
        file=st.file_uploader("upload a CSV file", type=["csv"])
            
            
        if file is not None:
            data=load_data(file)
            
            
            # KNN Imputation technique
            a = data.drop('LET_IS', axis=1)
            b = data['LET_IS'] 
            imputer = KNNImputer(n_neighbors=5)
            a = imputer.fit_transform(a)
            prediction = final_model.predict(a)
            st.write(data.head())
            # Define the data
            
            data['Classifications']=final_model.predict(a)
            
            Classifications= data['Classifications']
            
            
            # Generate the HTML representation of the DataFrame
            df = data['Classifications']
        
            
            # Step 1: Select the type of input
            
            input_type = st.selectbox("Select the input type", ["Series","Perticular","Random", "Range"])   
            
            # Step 2: Handle input based on the selected type
            l=df.apply(lambda x:'unknown (alive)' if x==0 else 'cardiogenic shock' if x==1 else 'pulmonary edema' if x==2 else 'myocardial rupture' if x==3 else 'progress of congestive heart failure' if x==4 else 'thromboembolism' if x==5 else 'asystole' if x==6 else 'ventricular fibrillation' if x==7 else"")
            
            
            if input_type == "Series":
                num_values = st.number_input("Enter the Starting number of prediction",min_value=0,step=1)
                st.write(l.iloc[num_values:])
            elif input_type == "Perticular":
                num_values = st.number_input("Enter the Selective ID for prediction",min_value=0,step=1)
                st.write(l.iloc[num_values])
            elif input_type == "Random":
                num_values = st.number_input("Enter the number of values",min_value=0,step=1)
                st.write(l.sample(num_values))
            else:
                start = st.number_input("Enter the start value", min_value=0,step=1)
                end = st.number_input("Enter the end value", min_value=0,step=1)
                st.write(l.iloc[start:end] )
                
            csv_data = df.to_csv(index=False)
            b64=base64.b64encode(csv_data.encode()).decode()
            href=f'<a href="data:file/csv;base64,{b64}" download="prediction.csv">Download CSV</a>'
            st.markdown(href,unsafe_allow_html=True)
    

    
