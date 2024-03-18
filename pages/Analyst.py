import folium
import requests
import pandas as pd
import seaborn as sns
import streamlit as st
import plotly.express as px
import matplotlib.pyplot as plt
from streamlit_folium import folium_static




# 네비게이션 구성 설정
st.set_page_config(initial_sidebar_state='collapsed', layout='wide')

# 색깔 가져오기


# main페이지 변수 할당
if 'data' in st.session_state:
    numric_features = st.session_state.numric_features
    categorical = st.session_state.categorical
    target = st.session_state.target
    data = st.session_state.data
    colors = st.session_state.colors

else:
    st.switch_page("Main.py")


# CSS로 버튼 center
################################################################################
# CSS 스타일을 포함한 HTML 문자열
centered_button_style = """
<style>
    .stButton>button {
        display: block;
        margin: auto;
        border: None;
        box-shadow: 2px 2px 5px rgba(0, 0, 0, 0.3);
    }
    .stButton>button:hover {
        background-color: whiteblue !important;
        color: red !important;
        border: 0.2px solid #E5E5E5 !important;
        box-shadow: None !important;
        font-weight: bold;
        transition: box-shadow 0.3s ease;
    }
    #root > div:nth-child(1) > div.withScreencast > div > div > div > section.main.st-emotion-cache-uf99v8.ea3mdgi8 > div.block-container.st-emotion-cache-z5fcl4.ea3mdgi5 > div > div > div > div.st-emotion-cache-ocqkz7.e1f1d6gn5{
         position: fixed;
         top: 2.9em;
         width: 92%;
         z-index: 9999;
         background-color: white;
    }
    
    # #root > div:nth-child(1) > div.withScreencast > div > div > div > section.main.st-emotion-cache-uf99v8.ea3mdgi8 > div.block-container.st-emotion-cache-z5fcl4.ea3mdgi5 > div > div > div > div.st-emotion-cache-ocqkz7.e1f1d6gn5 > div:nth-child(1) > div > div > div > div > div{
    #      margin-left: 1em;
    #      # width: 5em;
    #      # height: 5em;
    #      # border-radius: 50%;
    }
    #root > div:nth-child(1) > div.withScreencast > div > div > div > section.main.st-emotion-cache-uf99v8.ea3mdgi8 > div.block-container.st-emotion-cache-z5fcl4.ea3mdgi5 > div > div > div > div.st-emotion-cache-ocqkz7.e1f1d6gn5 > div:nth-child(1) > div > div > div > div > div > button{
        width: 8em;
        margin-left: 0em;
    }
    
    #root > div:nth-child(1) > div.withScreencast > div > div > div > section.main.st-emotion-cache-uf99v8.ea3mdgi8 > div.block-container.st-emotion-cache-z5fcl4.ea3mdgi5 > div > div > div > div.st-emotion-cache-ocqkz7.e1f1d6gn5 > div:nth-child(2) > div > div > div > div > div > button{
        width: 75%;
        margin-right: 0em;
        position: absolute;
        right: 0em;
    }
    
    #root > div:nth-child(1) > div.withScreencast > div > div > div > section.main.st-emotion-cache-uf99v8.ea3mdgi8 > div.block-container.st-emotion-cache-z5fcl4.ea3mdgi5 > div > div > div > div.st-emotion-cache-ocqkz7.e1f1d6gn5 > div:nth-child(3) > div > div > div > div > div > button{
        width: 75%;
        margin-right: 0em;
    }
</style>
"""


# CSS를 사용하여 버튼을 가운데로 정렬
st.markdown(centered_button_style, unsafe_allow_html=True)
################################################################################




# 페이지 구성
col1, col2, col3 = st.columns(3)

with col1:
    if st.button("Main"):
        st.switch_page("Main.py")
with col2:
    if st.button("Analyst"):
        st.switch_page("pages/Analyst.py")
with col3:
    if st.button("Investor"):
        st.switch_page("pages/Investor.py")

# ####
# st.divider()
# ####
def tab1_num():
    st.session_state.tab1_cat = None

def tab1_cat():
    st.session_state.tab1_num = None



st.markdown('# Insights - Analyst')
tab1, tab2, tab3, tab4 = st.tabs(["Info","Single Features Select", "Multiple Features Select", "Map(addr_state)"])
with tab1:
    col1, col2 = st.columns([0.4,0.6])
    with col1:
        # 데이터프레임에서 레이블 및 값 가져오기
        st.markdown('### Loan Status Ratio')
        values = data[target].value_counts()
        df = pd.DataFrame.from_dict({"class": ["Fully Paid", "Charged Off"], "values": values})
        fig = px.pie(df, values='values', names='class',color_discrete_sequence=colors)
        st.plotly_chart(fig, theme="streamlit",use_container_width=True)
        st.write('Total Count',f'{len(data):,}')
        # with st.container():
            # st.metric('Total Count',f'{len(data):,}')
            # st.markdown(f'Total :  **{len(data):,}**')
    
    with col2:
        st.markdown('### Data Frame')
        st.dataframe(data)
    ####
    st.divider()
with tab2:
    col1, col2, col3 = st.columns(3)
    with col1:
        single_select_feature_num = st.selectbox('특성 선택', numric_features, label_visibility='collapsed', index=None, placeholder="수치형 특성 선택", key='tab1_num', on_change=tab1_num)
            
    with col2:
        single_select_feature_cat = st.selectbox('특성 선택', categorical, label_visibility='collapsed', index=None, placeholder="범주형 특성 선택", key='tab1_cat', on_change=tab1_cat)
    with col3:
        bins = st.slider('bins', min_value=0, max_value=100, value=10, label_visibility='collapsed', disabled=False if single_select_feature_num else True)
        col3_1, col3_2 = st.columns(2)
        with col3_1:
            hue_flag = st.checkbox('hue_flag', value=False if single_select_feature_cat else True, key='tab1_hue')
        with col3_2:
            kde_flag = st.checkbox('kde_flag', value=False if single_select_feature_cat else True, disabled=False if single_select_feature_num else True)
            
    col1, col2 = st.columns([0.95,0.05])
    with col1:
        if single_select_feature_num != None:
            fig = plt.figure(figsize=(15,4))
            sns.histplot(x=single_select_feature_num, hue=target if hue_flag else None, multiple='stack',bins="auto" if not bins else bins,kde=kde_flag, data=data,palette=colors)
            st.pyplot(fig)
            fig = None
        if single_select_feature_cat != None:
            fig = plt.figure(figsize=(15,4))
            if single_select_feature_cat == 'addr_state' or single_select_feature_cat == 'sub_grade':
                plt.yticks(fontsize=5)  # 폰트 크기 조절
            sns.histplot(y=single_select_feature_cat, hue=target if hue_flag else None, multiple='stack', shrink=0.5, data=data,palette=colors)
            st.pyplot(fig)
            fig = None
        with col2:
            pass

        
with tab3:
    st.markdown('### Feature Select Multiple')
    col1, col2, col3 = st.columns(3)
    with col1:
        select_feature_num1 = st.selectbox('특성 선택', numric_features, label_visibility='collapsed', index=None, placeholder="수치형 특성 선택", key='tab2_num1')
            
    with col2:
        select_feature_num2 = st.selectbox('특성 선택', numric_features, label_visibility='collapsed', index=None, placeholder="수치형 특성 선택", key='tab2_num2')
    with col3:
        if (select_feature_num1 != None) & (select_feature_num2 != None):
            if select_feature_num1 == select_feature_num2:
                st.metric('Coefficient','1.0')
            else:
                st.metric('Coefficient',f'{round(data[[select_feature_num1,select_feature_num2]].corr().loc[select_feature_num1,select_feature_num2],3):,}')
        else:
            st.metric('Coefficient','0.0')
        hue_flag = st.checkbox('hue_flag', value=False, key='tab2_hue')

    col1, col2 = st.columns([0.95,0.05])
    with col1:
        if (select_feature_num1 != None) & (select_feature_num2 != None):
            fig = plt.figure(figsize=(15,4))
            sns.scatterplot(x=select_feature_num1,y=select_feature_num2, data=data,palette=colors,alpha=0.3, hue=target if hue_flag else None)
            st.pyplot(fig)
            fig = None
    with col2:
        pass
            
# last_fico_range_high
with tab4:
    # copy_data = data.copy()
    copy_data = data
    charged_off_data = (copy_data[["addr_state", "loan_status"]].groupby("addr_state").loan_status.value_counts(normalize=True).loc[pd.IndexSlice[:, "Charged Off"]] * 100).reset_index().set_axis(["addr_state", "proportion"], axis=1)
    last_fico_score_data = copy_data[["addr_state", "last_fico_range_low", "last_fico_range_high"]].groupby("addr_state").mean().mean(axis=1).reset_index().set_axis(["addr_state", "last_fico_score"], axis=1)
    dti_data = copy_data[["addr_state", "dti"]].groupby("addr_state").mean().reset_index()
    int_rate_data = copy_data[["addr_state", "int_rate"]].groupby("addr_state").mean().reset_index()
    loan_amnt_data = copy_data[["addr_state", "loan_amnt"]].groupby("addr_state").mean().reset_index()
    
    state_geo = requests.get("https://raw.githubusercontent.com/python-visualization/folium-example-data/main/us_states.json").json() # US states
    state_geo.get("features").append({"type": "Feature", 
                                      "id": "DC", 
                                      "properties": {"name": "District of Columbia"},
                                      "geometry": {"type": "Polygon",
                                                   "coordinates": [[[-77.035264,38.993869], 
                                                                    [-77.117418,38.933623], 
                                                                    [-77.040741,38.791222], 
                                                                    [-76.909294,38.895284],
                                                                    [-77.035264,38.993869]]]}})
    
    m = folium.Map(location=[48, -102], tiles=folium.TileLayer("cartodbpositron", name="base_map"), zoom_start=3, zoom_control=False)
    
    charged_off_chor = folium.Choropleth(geo_data=state_geo, 
                                         data=charged_off_data, 
                                         columns=charged_off_data.columns, 
                                         key_on="feature.id", 
                                         fill_color="OrRd", 
                                         line_weight=0.7, 
                                         line_opacity=0.4, 
                                         name="Charged Off 비율", 
                                         legend_name="Charged Off 비율", 
                                         show=False)
    
    for data in charged_off_chor.geojson.data.get("features"):
        data.get("properties")["id"] = f"{data.get('properties').get('name')} ({data.get('id')})"
        data.get("properties")["charged_off_rate"] = f"{charged_off_data[charged_off_data.addr_state == data.get('id')]['proportion'].values.round(2)[0]}%"
    folium.GeoJsonTooltip(fields=["id", "charged_off_rate"], aliases=["주 이름", "Charged Off 비율"], style="font-size: 12px").add_to(charged_off_chor.geojson)
    
    
    last_fico_score_chor = folium.Choropleth(geo_data=state_geo, 
                                             data=last_fico_score_data, 
                                             columns=last_fico_score_data.columns, 
                                             key_on="feature.id", 
                                             fill_color="OrRd", 
                                             line_weight=0.7, 
                                             line_opacity=0.4, 
                                             name="평균 최근 신용점수", 
                                             legend_name="평균 최근 신용점수", 
                                             show=False)
    
    for data in last_fico_score_chor.geojson.data.get("features"):
        data.get("properties")["id"] = f"{data.get('properties').get('name')} ({data.get('id')})"
        data.get("properties")["last_fico_score"] = last_fico_score_data[last_fico_score_data.addr_state == data.get("id")]["last_fico_score"].values.round()[0]
    folium.GeoJsonTooltip(fields=["id", "last_fico_score"], aliases=["주 이름", "평균 최근 신용점수"], style="font-size: 12px").add_to(last_fico_score_chor.geojson)
    
    
    dti_chor = folium.Choropleth(geo_data=state_geo, 
                                 data=dti_data, 
                                 columns=dti_data.columns, 
                                 key_on="feature.id", 
                                 fill_color="OrRd", 
                                 line_weight=0.7, 
                                 line_opacity=0.4, 
                                 name="평균 dti", 
                                 legend_name="평균 dti", 
                                 show=False)
    
    for data in dti_chor.geojson.data.get("features"):
        data.get("properties")["id"] = f"{data.get('properties').get('name')} ({data.get('id')})"
        data.get("properties")["dti"] = f"{dti_data[dti_data.addr_state == data.get('id')]['dti'].values.round(2)[0]}%"
    folium.GeoJsonTooltip(fields=["id", "dti"], aliases=["주 이름", "평균 dti"], style="font-size: 12px").add_to(dti_chor.geojson)
    
    
    int_rate_chor = folium.Choropleth(geo_data=state_geo, 
                                      data=int_rate_data, 
                                      columns=int_rate_data.columns, 
                                      key_on="feature.id", 
                                      fill_color="OrRd", 
                                      line_weight=0.7, 
                                      line_opacity=0.4, 
                                      name="평균 이자율", 
                                      legend_name="평균 이자율", 
                                      show=False)
    
    for data in int_rate_chor.geojson.data.get("features"):
        data.get("properties")["id"] = f"{data.get('properties').get('name')} ({data.get('id')})"
        data.get("properties")["int_rate"] = f"{int_rate_data[int_rate_data.addr_state == data.get('id')]['int_rate'].values.round(2)[0]}%"
    folium.GeoJsonTooltip(fields=["id", "int_rate"], aliases=["주 이름", "평균 이자율"], style="font-size: 12px").add_to(int_rate_chor.geojson)


    loan_amnt_chor = folium.Choropleth(geo_data=state_geo, 
                                       data=loan_amnt_data, 
                                       columns=loan_amnt_data.columns, 
                                       key_on="feature.id", 
                                       fill_color="OrRd", 
                                       line_weight=0.7, 
                                       line_opacity=0.4, 
                                       name="평균 대출금", 
                                       legend_name="평균 대출금", 
                                       show=False)
    
    for data in loan_amnt_chor.geojson.data.get("features"):
        data.get("properties")["id"] = f"{data.get('properties').get('name')} ({data.get('id')})"
        data.get("properties")["loan_amnt"] = f"$ {loan_amnt_data[loan_amnt_data.addr_state == data.get('id')]['loan_amnt'].values.round(2)[0]:,}"
    folium.GeoJsonTooltip(fields=["id", "loan_amnt"], aliases=["주 이름", "평균 대출금"], style="font-size: 12px").add_to(loan_amnt_chor.geojson)

    
    charged_off_chor.add_to(m)
    last_fico_score_chor.add_to(m)
    dti_chor.add_to(m)
    int_rate_chor.add_to(m)
    loan_amnt_chor.add_to(m)
    
    folium.LayerControl(collapsed=False).add_to(m)
    folium_static(m, width=1200,height=500)
    ####
    st.divider()
    ####
