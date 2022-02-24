import streamlit as st
import pickle
import sklearn
from streamlit_tags import st_tags
import pandas as pd
import itertools
import base64
hide_menu_style = """
        <style>
        #MainMenu {visibility: hidden;}
        </style>
        """
st.markdown(hide_menu_style, unsafe_allow_html=True)

#loading modulus model
filename1 = 'rubbermodulus_mlpr_rand_model.sav'
modulus_model = pickle.load(open(filename1, 'rb'))

#loading tensile model
filename2 = 'tensile_model_rubber2_rf_gridopt.sav'
tensile_model = pickle.load(open(filename2, 'rb'))

#loading volume_fraction model
filename3 = 'rubber_vfra_randopt.sav'
vfraction_model = pickle.load(open(filename3, 'rb'))

# lading saved model
filename = 'rubber2_rf_gridopt.sav'
# filename = 'rubber_xgb_randopt2.sav'
loaded_model1 = pickle.load(open(filename, 'rb'))
filename = 'num_pipeline_in.sav'
loaded_pipeline = pickle.load(open(filename, 'rb'))
st.header('Rubber Compound Optimizer')
st.markdown(""" Welcome, Enter the values for Tensile stregth prediction and selecting optimum compound combination
Kindly note that this model is optimized using NBR values. """)


@st.cache
def filedownload(df):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()  # strings <-> bytes conversions
    href = f'<a href="data:file/csv;base64,{b64}" download="rubber optimum.csv">Download CSV File</a>'
    return href


def user_input():
    x1,x2,x3,x4,x5,x6=[],[],[],[],[],[]

    xl1 = st_tags(label='Enter values for Dose (kGy) between (0-200):', value=[100.0],maxtags=-1,key="alnf2")
    xl2 = st_tags(label='Enter values for Sensitizer (phr) between (0-4):', value=[1.0], maxtags=-1,key="alnf3")
    xl3 = st_tags(label='Enter values for Filler (phr) between (0-60):', value=[10.0,20.0], maxtags=-1,key="alnf4")
    xl4 = st_tags(label='Enter values for Antioxidant (phr) between (0-2):', value=[1.0], maxtags=-1,key="alnf5")
    xl5 = st_tags(label='Enter values for Accelerator (phr) between (0-1.5):',value=[1.5], maxtags=-1,key="alnf6")
    xl6 = st_tags(label='Enter values for Sulfur (phr) between (0.05-1.5):',value=[1.0], maxtags=-1,key="alnf7")

    try:
        for value in xl1:
            valuef = float(value)
            x1.append(valuef)

        for value in xl2:
            valuef = float(value)
            x2.append(valuef)

        for value in xl3:
            valuef = float(value)
            x3.append(valuef)

        for value in xl4:
            valuef = float(value)
            x4.append(valuef)

        for value in xl5:
            valuef = float(value)
            x5.append(valuef)

        for value in xl6:
            valuef = float(value)
            x6.append(valuef)
    except:
        st.write(' Error...Enter valid numbers')

    x_values=[x1,x2,x3,x4,x5,x6]
    return x_values


st.sidebar.header('Estimate Compound Cost')

ru_price = st.sidebar.number_input("Enter NBR price", value=4.0)
se_price = st.sidebar.number_input("Enter Sensitizer(TMPTA) price", value=5.0)
fi_price = st.sidebar.number_input("Enter Filler price", value=0.68)
ao_price = st.sidebar.number_input("Enter Antioxidant price", value=2.0)
ac_price = st.sidebar.number_input("Enter Accelerator price", value=1.59)
su_price = st.sidebar.number_input("Enter Sulfur price", value=0.6)

x_values1 = user_input()
x_values_list = list(itertools.product(*x_values1))

# sidebar estimate cost
if st.button('Calculate'):
    try:
        x_values_list_tr = loaded_pipeline.transform(x_values_list)
        df7 = pd.DataFrame(x_values1)
        df8 = df7.transpose()
        df8.columns = ["Dose (kGy)", 'Sensitizer (phr)', 'Filler (phr)', 'Antioxidant (phr)', 'Accelerator (phr)',
                       'Sulfur (phr)']
        st.write('Entered value combinations')
        st.dataframe(df8)

        modulus_values = []
        tensile_values = []
        vfraction_values = []

        y_vlaues = []
        for xv in x_values_list_tr:
            m_value = modulus_model.predict([xv])
            modulus_values.append(m_value)
            t_value = tensile_model.predict([xv])
            tensile_values.append(t_value)
            v_value = vfraction_model.predict([xv])
            vfraction_values.append(v_value)


        # sorting results
        #st.write('values predicted')
        #st.write(modulus_values)
        #st.write(tensile_values)
        #st.write(vfraction_values)
        df4 = pd.DataFrame({'Modulus Predicted': modulus_values, 'Tensile Predicted': tensile_values, 'V_Fraction Predicted':vfraction_values})

        #df4 = pd.DataFrame(modulus_values,tensile_values,vfraction_values, columns=['Modulus Predicted', 'Tensile Predicted', 'V_Fraction Predicted'])
        #st.write('df4 calculated')
        #st.dataframe(df4)
        df3 = pd.DataFrame(x_values_list, columns=["Dose (kGy)", 'Sensitizer (phr)', 'Filler (phr)', 'Antioxidant (phr)',
                                                   'Accelerator (phr)', 'Sulfur (phr)'])

        df5 = pd.concat([df3, df4], axis=1, join='outer')
        df5['Cost per KG'] = (100 * ru_price + df5['Sensitizer (phr)'] * se_price + df5['Filler (phr)'] * fi_price +
                              df5['Antioxidant (phr)'] * ao_price +
                              df5['Accelerator (phr)'] * ac_price + df5['Sulfur (phr)'] * su_price) / (
                                         100 + df5['Sensitizer (phr)'] +
                                         df5['Filler (phr)'] + df5['Antioxidant (phr)'] + df5['Accelerator (phr)'] +
                                         df5['Sulfur (phr)'])
        df6 = df5.sort_values('Tensile Predicted', ascending=False)
        df6 = df6.reset_index(drop=True)

        st.markdown(""" **** """)
        st.subheader('Results')
        st.dataframe(df6)

        st.markdown(filedownload(df6), unsafe_allow_html=True)

    except :
        st.write('Error')
        st.write('Enter values for all compounds...')






