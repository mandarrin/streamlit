import shap
import time
import base64
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
from streamlit_shap import st_shap
from streamlit_extras.let_it_rain import rain
from sklearn.base import BaseEstimator, TransformerMixin

class FeatureSelect(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_copy = X.copy()
        X_copy = X_copy.drop(["addr_state"], axis=1)
        return X_copy

class FeatureEngineer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_copy = X.copy()
        X_copy["term"] = X_copy.term.str.extract("(\d+)").astype(np.float64)
        X_copy["open_acc_rate"] = (X_copy.open_acc / X_copy.total_acc * 100).round(1)
        X_copy["last_fico_score"] = X_copy[["last_fico_range_low", "last_fico_range_high"]].mean(axis=1).round()
        X_copy["installment"] = np.vectorize(lambda x, y, z: round((x * y * 0.01 / 12) * ((1 + y * 0.01 / 12) ** z) / (((1 + y * 0.01 / 12) ** z) - 1), 2))(X_copy.loan_amnt, X_copy.int_rate, X_copy.term)
        X_copy = X_copy.drop(["open_acc", "total_acc", "last_fico_range_low", "last_fico_range_high"], axis=1)
        return X_copy

if "random" not in st.session_state:
    st.session_state.random = np.random.default_rng(seed=6)

if "predict" not in st.session_state:
    st.session_state.predict = None

if "predict_proba" not in st.session_state:
    st.session_state.predict_proba = None

# 네비게이션 구성 설정
st.set_page_config(initial_sidebar_state='collapsed', layout='wide')

# 데이터 캐싱
@st.cache_data
def get_model():
    model = joblib.load("model/final_model2.pkl")
    return model

def choice_random(data):
    random_data = data.loc[st.session_state.random.choice(data.index), :]
    st.session_state.loan_amnt = random_data.loan_amnt
    st.session_state.term = random_data.term
    st.session_state.int_rate = random_data.int_rate
    st.session_state.sub_grade = random_data.sub_grade
    st.session_state.verification_status = random_data.verification_status
    st.session_state.addr_state = random_data.addr_state
    st.session_state.dti = random_data.dti
    st.session_state.open_acc = random_data.open_acc
    st.session_state.revol_util = random_data.revol_util
    st.session_state.total_acc = random_data.total_acc
    st.session_state.last_fico_range_high = random_data.last_fico_range_high
    st.session_state.last_fico_range_low = random_data.last_fico_range_low
    st.session_state.avg_cur_bal = random_data.avg_cur_bal
    st.session_state.mo_sin_old_rev_tl_op = random_data.mo_sin_old_rev_tl_op
    st.session_state.mo_sin_rcnt_rev_tl_op = random_data.mo_sin_rcnt_rev_tl_op
    st.session_state.pct_tl_nvr_dlq = random_data.pct_tl_nvr_dlq
    st.session_state.colors = colors

# 데이터 캐싱
@st.cache_data
def example():
            rain(
                emoji='💧',
                font_size=48,
                falling_speed=1,
                animation_length=2,
            )
# , explainer
model = get_model()

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
         position: fixed !important;
         top: 2.9em !important;
         width: 92% !important;
         z-index: 9999 !important;
         background-color: white !important;
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
</style>
"""
random_result_button_style="""
<style>
    #tabs-bui19-tabpanel-4 > div > div > div > div.st-emotion-cache-ocqkz7.e1f1d6gn5 > div:nth-child(1) > div > div > div > div > div > butt{
        color: red;
    } 
</style>

"""

column_property="""
<style>
    .st-ae .st-emotion-cache-1yycg8b.e1f1d6gn3 > div > div > div > div > div{
        height: 5em;
        padding: 1.5em;
    }
    .st-ae .st-emotion-cache-1yycg8b.e1f1d6gn3 button{
        height:2em;
        width: 13em;
    }

</style>
"""


# CSS를 사용하여 버튼을 가운데로 정렬
st.markdown(centered_button_style, unsafe_allow_html=True)

# CSS를 사용하여 버튼을 가운데로 정렬
st.markdown(random_result_button_style, unsafe_allow_html=True)

# CSS를 사용하여 버튼을 가운데로 정렬
st.markdown(column_property, unsafe_allow_html=True)

################################################################################

#### 특성 내용 담는 변수
################################################################################
addr_state = """
대출 신청자가 대출 신청서에 기재한 주소의 상태를 나타냅니다. 대출 신청자의 거주 주소는 신용 리스크와 관련된 지역적 요인을 고려하는 데 사용될 수 있습니다. 예를 들어, 특정 지역의 경기 상황이나 부동산 시장의 안정성은 대출자의 상환 능력에 영향을 줄 수 있습니다.
"""
last_fico_range_high = """
대출자의 최근 FICO 신용 점수 범위의 상한을 나타냅니다. 최근 FICO 신용 점수는 대출자의 현재 신용 신뢰성을 반영합니다. 높은 FICO 신용 점수는 대출자의 신용 신뢰성을 나타내며, 낮은 신용 위험을 의미할 수 있습니다.
"""
last_fico_range_low = """
대출자의 최근 FICO 신용 점수 범위의 하한을 나타냅니다. 최근 FICO 신용 점수는 대출자의 현재 신용 신뢰성을 반영합니다. 낮은 FICO 신용 점수는 대출자의 신용 신뢰성을 나타내며, 높은 신용 위험을 의미할 수 있습니다.
"""
dti = """
대출자의 월간 부채 지불액을 월간 소득에 대한 비율로 나타냅니다. 부채 비율은 대출자의 부채 상환 능력을 평가하는 데 사용됩니다. 높은 부채 비율은 대출자가 부채를 상환하는 데 어려움을 겪을 수 있음을 나타낼 수 있습니다. 일반적으로 부채 비율이 낮을수록 대출 신청자의 신용 위험은 낮아질 수 있습니다.
"""
verification_status = """
대출 신청자의 소득이 Lending Club에 의해 확인되었는지 여부, 확인되지 않았는지 또는 소득 출처가 확인되었는지 여부를 나타냅니다. 
소득 확인은 대출 신청자의 신용평가 및 대출 상환 능력에 중요한 영향을 미칩니다. 소득 확인 여부는 대출 신청자의 신용 위험을 평가하고 대출 상환 능력을 판단하는 데 사용됩니다. 
소득이 확인될수록 대출자의 신용 프로파일이 강화되고 대출 상환 능력이 높아질 수 있습니다.
"""
### 리볼빙
revol_util="""
대출자의 리볼빙 대출 이용률을 나타냅니다. 리볼빙 대출 이용률은 대출자가 리볼빙 대출 한도에 대해 현재 사용한 비율을 나타내며, 대출자의 신용 활동 수준과 대출 상환 능력을 평가하는 데 사용될 수 있습니다.
"""
mo_sin_old_rev_tl_op="""
가장 오래된 리볼빙 계정 개설 이후 경과한 개월 수를 나타냅니다. 이 값은 대출자의 신용 이용 패턴과 신용 활동에 영향을 줄 수 있습니다. 가장 오래된 리볼빙 계정 개설 이후 경과한 개월 수가 길수록 대출자의 신용 이용이 안정적이다는 것을 나타낼 수 있습니다.
"""
mo_sin_rcnt_rev_tl_op="""
가장 최근의 리볼빙 계정 개설 이후 경과한 개월 수를 나타냅니다. 이 값은 대출자의 신용 이용 패턴과 신용 활동에 영향을 줄 수 있습니다. 가장 최근의 리볼빙 계정 개설 이후 경과한 개월 수가 길수록 대출자의 신용 이용이 안정적이다는 것을 나타낼 수 있습니다.
"""
open_acc="""
대출자의 신용 보고서에 기록된 현재 개설된 대출 수를 나타냅니다. 개설된 대출 수는 대출자의 신용 활동 수준과 대출 상환 능력을 평가하는 데 사용될 수 있습니다.
"""
total_acc="""
대출자의 현재 신용 계정 수를 나타냅니다. 이 값은 대출자의 신용 이용 및 신용 프로파일에 영향을 줄 수 있습니다. 높은 신용 계정 수는 대출자의 신용 이용이 다양하며 신용 이용 가능성이 높다는 것을 나타낼 수 있습니다.
"""
avg_cur_bal="""
대출자의 모든 계정의 평균 현재 잔액을 나타냅니다. 이 값은 대출자의 재정 상태와 신용 이용에 영향을 줄 수 있습니다. 모든 계정의 평균 현재 잔액이 높을수록 대출자의 신용 상태가 좋다는 것을 나타낼 수 있습니다.
"""
pct_tl_nvr_dlq="""
대출자의 연체 없는 거래 비율을 나타냅니다. 이 값은 대출자의 신용 상태와 상환 능력을 평가하는 데 도움을 줄 수 있습니다. 높은 비율은 대출자가 거래를 신뢰할 수 있고 상환 불이행 가능성이 낮다는 것을 나타낼 수 있습니다.
"""
loan_amnt="""
대출자가 신청한 대출의 금액을 나타냅니다. 이 값은 대출자의 자금 요구와 대출 상환 능력에 영향을 줄 수 있습니다. 대출 신청액이 크다는 것은 대출자가 높은 금액의 자금을 필요로 한다는 것을 나타낼 수 있습니다.
"""
term="""
대출자가 상환 계약에서 계약 월 대출 상환금액보다 적은 금액을 상환해야 하는 예상 기간을 나타냅니다. 상환 유예 기간은 대출자가 얼마나 오랜 기간 동안 낮은 상환금을 지불할 수 있는지를 나타내며, 대출자의 재정 상태를 평가하는 데 사용될 수 있습니다.
"""
int_rate="""
대출의 이자율을 나타냅니다. 이 값은 대출의 비용과 대출자의 상환 부담에 영향을 줄 수 있습니다. 대출 이자율이 높다는 것은 대출의 비용이 상대적으로 높다는 것을 나타낼 수 있습니다.
"""
sub_grade="""
Lending Club에서 지정한 대출 등급(subgrade)을 나타냅니다. 대출 등급은 대출자의 신용 위험을 나타내는 중요한 지표입니다. 높은 등급은 대출자의 신용 상태와 상환 능력이 우수하다는 것을 나타낼 수 있습니다.
"""
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

####
# st.divider()
####

def stream_data(text):
    for word in text.split():
        yield word + " "
        time.sleep(0.01)

st.markdown('# Insights - Investor')

tab1, tab2, tab3, tab4, tab5 = st.tabs(['Step1(privacy)','Step2(revol)','Step3(credit)','Step4(loan)','Result'])
with tab1:
    col1, col2 = st.columns(2)
    with col1:
        st.markdown('### Feature Select')
        st.selectbox("verification_status", options=["Not Verified", "Source Verified", "Verified"], key="verification_status")
        # st.selectbox("addr_state", options=data.addr_state.unique(), key="addr_state")
        st.slider("last_fico_range_high", min_value=300, max_value=850,step=1, key="last_fico_range_high")
        st.slider("last_fico_range_low", min_value=300, max_value=850,step=1, key="last_fico_range_low")
        st.slider("dti", min_value=0.00, max_value=40.00,step=0.01, key="dti")
    with col2:
        st.markdown('### Feature Property')
        # with st.expander('Privacy(개인정보)'): # 6
        col1, col2 = st.columns([0.4,0.6])
        with col1:
            f = st.button("verification_status")
            # a = st.button("emp_length")
            # b = st.button("addr_state")
            c = st.button("last_fico_range_high")
            d = st.button("last_fico_range_low")
            e = st.button("dti")
        with col2:
            # if a:
            #     st.write_stream(stream_data(emp_length))
            # if b:
            #     st.write_stream(stream_data(addr_state))
            if c:
                st.write_stream(stream_data(last_fico_range_high))
            if d:
                st.write_stream(stream_data(last_fico_range_low))
            if e:
                st.write_stream(stream_data(dti))
            if f:
                st.write_stream(stream_data(verification_status))

with tab2:
    col1, col2 = st.columns(2)
    with col1:
        st.markdown('### Feature Select')
        st.slider("revol_util", min_value=0.0, max_value=100.0,step=0.1, key="revol_util")
        st.slider("mo_sin_old_rev_tl_op", min_value=0, max_value=1000,step=1, key="mo_sin_old_rev_tl_op")
        st.slider("mo_sin_rcnt_rev_tl_op", min_value=0, max_value=500,step=1, key="mo_sin_rcnt_rev_tl_op")
    with col2:
        st.markdown('### Feature Property')
        col1, col2 = st.columns([0.4,0.6])
        with col1:
            a = st.button("revol_util")
            b = st.button("mo_sin_old_rev_tl_op")
            c = st.button("mo_sin_rcnt_rev_tl_op")
        with col2:
            if a:
                st.write_stream(stream_data(revol_util))
            if b:
                st.write_stream(stream_data(mo_sin_old_rev_tl_op))
            if c:
                st.write_stream(stream_data(mo_sin_rcnt_rev_tl_op))
with tab3:
    col1, col2 = st.columns(2)
    with col1:
        st.markdown('### Feature Select')
        st.slider("open_acc", min_value=0, max_value=500, key="open_acc")
        st.slider("total_acc", min_value=1, max_value=177, key="total_acc")
        st.slider("avg_cur_bal", min_value=0, max_value=1000000, key="avg_cur_bal")
        st.slider("pct_tl_nvr_dlq", min_value=0.0, max_value=100.0,step=0.1, key="pct_tl_nvr_dlq")
    with col2:
        st.markdown('### Feature Property')
        col1, col2 = st.columns([0.4,0.6])
        with col1:
            a = st.button("open_acc")
            b = st.button("total_acc")
            c = st.button("avg_cur_bal")
            d = st.button("pct_tl_nvr_dlq")
        with col2:
            if a:
                st.write_stream(stream_data(open_acc))
            if b:
                st.write_stream(stream_data(total_acc))
            if c:
                st.write_stream(stream_data(avg_cur_bal))
            if d:
                st.write_stream(stream_data(pct_tl_nvr_dlq))
with tab4:
    col1, col2 = st.columns(2)
    with col1:
        st.markdown('### Feature Select')
        st.selectbox("term", options=[" 36 months", " 60 months"], key="term")
        st.slider("loan_amnt", min_value=1000, max_value=40000, key="loan_amnt")
        st.slider("int_rate", min_value=5.00, max_value=31.00,step=0.01, key="int_rate")
        st.selectbox("sub_grade",  options=sorted(data.sub_grade.unique()), key="sub_grade")
    with col2:
        st.markdown('### Feature Property')
        col1, col2 = st.columns([0.4,0.6])
        with col1:
            b = st.button("term")
            a = st.button("loan_amnt")
            c = st.button("int_rate")
            d = st.button("sub_grade")
        with col2:
            if a:
                st.write_stream(stream_data(loan_amnt))
            if b:
                st.write_stream(stream_data(term))
            if c:
                st.write_stream(stream_data(int_rate))
            if d:
                st.write_stream(stream_data(sub_grade))
with tab5:
    
    # predict_data = pd.DataFrame(data=[[st.session_state.loan_amnt, 
    #                                    st.session_state.term, 
    #                                    st.session_state.int_rate, 
    #                                    st.session_state.sub_grade, 
    #                                    st.session_state.verification_status, 
    #                                    np.nan, 
    #                                    st.session_state.dti, 
    #                                    st.session_state.open_acc, 
    #                                    st.session_state.revol_util, 
    #                                    st.session_state.total_acc, 
    #                                    st.session_state.last_fico_range_high, 
    #                                    st.session_state.last_fico_range_low, 
    #                                    st.session_state.avg_cur_bal, 
    #                                    st.session_state.mo_sin_old_rev_tl_op, 
    #                                    st.session_state.mo_sin_rcnt_rev_tl_op, 
    #                                    st.session_state.pct_tl_nvr_dlq]], 
    #                             columns=["loan_amnt", "term", "int_rate", "sub_grade", 
    #                                      "verification_status", "addr_state", "dti", "open_acc", 
    #                                      "revol_util", "total_acc", "last_fico_range_high", "last_fico_range_low", 
    #                                      "avg_cur_bal", "mo_sin_old_rev_tl_op", "mo_sin_rcnt_rev_tl_op", "pct_tl_nvr_dlq"])
        
    # predict_data_preprocessed = model[:4].transform(predict_data)
    # shap_value = shap.TreeExplainer(model[-1], feature_perturbation="tree_path_dependent").shap_values(predict_data_preprocessed)[1]
    #         shap_value = np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
    #     st.session_state.predict = model.predict(predict_data)
    #     predict = st.session_state.predict
    #     predict_proba = model.predict_proba(predict_data)
    # last_model_element를 사용하여 계산 진행
    # predict_data_preprocessed = model[:4].transform(predict_data)
    # shap_value = shap.TreeExplainer(model[-1], feature_perturbation="tree_path_dependent").shap_values(predict_data_preprocessed)[1]
    
    # st.session_state.predict = model.predict(predict_data)
    # predict = st.session_state.predict
    # predict_proba = model.predict_proba(predict_data)
        
    
    col1, col2 = st.columns([0.15,0.85],gap='small')
    with col1:
        col1, col2 = st.columns(2,gap='small')
        with col1:
            st.button("Random", help="데이터에서 랜덤하게 1행을 뽑습니다.", on_click=choice_random, args=(data, ))
        with col2:
            restul_button = st.button('predict', help="예측 결과를 보여줍니다.")
    with col2:
        pass

    col11, col22 = st.columns([0.5,0.5],gap='small')
    with col11:
        col1, col2 = st.columns(2,gap='small')
        with col1:
            st.markdown("<center><h2>repayment forecast results</h2></center>", unsafe_allow_html=True)
            if restul_button:
                predict_data = pd.DataFrame(data=[[st.session_state.loan_amnt, 
                                       st.session_state.term, 
                                       st.session_state.int_rate, 
                                       st.session_state.sub_grade, 
                                       st.session_state.verification_status, 
                                       np.nan, 
                                       st.session_state.dti, 
                                       st.session_state.open_acc, 
                                       st.session_state.revol_util, 
                                       st.session_state.total_acc, 
                                       st.session_state.last_fico_range_high, 
                                       st.session_state.last_fico_range_low, 
                                       st.session_state.avg_cur_bal, 
                                       st.session_state.mo_sin_old_rev_tl_op, 
                                       st.session_state.mo_sin_rcnt_rev_tl_op, 
                                       st.session_state.pct_tl_nvr_dlq]], 
                                columns=["loan_amnt", "term", "int_rate", "sub_grade", 
                                         "verification_status", "addr_state", "dti", "open_acc", 
                                         "revol_util", "total_acc", "last_fico_range_high", "last_fico_range_low", 
                                         "avg_cur_bal", "mo_sin_old_rev_tl_op", "mo_sin_rcnt_rev_tl_op", "pct_tl_nvr_dlq"])
                st.session_state.predict = model.predict(predict_data)
                st.session_state.predict_proba = model.predict_proba(predict_data)
                st.markdown("<br><br><br><br><center><h1><font color='red'>Charged Off</font></h1></center>" if st.session_state.predict else "<br><br><br><br><center><h1><font color='blue'>Fully Paid</font></h1></center>", unsafe_allow_html=True)
                if not st.session_state.predict:
                    st.balloons()
                else:
                    example()
            else:
                st.markdown("")
        with col2:
            with open("img/model.png", "rb") as image_file:
                image_base64 = base64.b64encode(image_file.read()).decode("utf-8")

            html_content = f"""
            <style>
            .tooltip {{
              position: realtive;
              display: inline-block;
              cursor: pointer;
            }}
            
            .tooltip .tooltiptext {{
              visibility: hidden;
              width: 100px;
              # background-color: black;
              color: white;
              # text-align: center;
              # border-radius: 6px;
              # padding: 5px 0;
              position: absolute;
              z-index: 1;
              # top: 3em;
              bottom: -27em;
              # left: 50%;
              # margin-left: -60px;
              opacity: 0;
              transition: opacity 0.3s;
            }}
            
            .tooltip:hover .tooltiptext {{
              visibility: visible;
              opacity: 1;
            }}
            </style>
            
            <div class="tooltip">
              <p>ℹ️</p>
              <span class="tooltiptext">
                                        <img src="data:image/png;base64,{image_base64}" style="width:750px;height:750px;">
              </span>
            </div>
            """
            
            st.markdown(html_content, unsafe_allow_html=True)
    with col22:
        st.markdown('## Select Features Value')
        if restul_button:
            colors = ['#66b3ff','#ff9999']
            colors2 = {
                        "Fully Paid": "#66b3ff",
                        "Charged Off": "#ff9999",}
            colors_list = list(colors2.values())  # 색상 값 목록 얻기
            
            df = pd.DataFrame.from_dict({"class": ["Fully Paid", "Charged Off"], "probability": st.session_state.predict_proba.flatten()})
            fig = px.pie(df, values='probability', names='class', title='probability',color='class',color_discrete_sequence=colors, category_orders={'class': ['Fully Paid', 'Charged Off']}) #color_discrete_sequence=colors[::-1]
            st.plotly_chart(fig, theme="streamlit",use_container_width=True)
        else:
            st.write('')

    # st.divider()
    st.markdown('### Waterfall Plot')
    with st.expander(''):
        col1, col2 = st.columns([0.3,0.7],gap='small')
        if restul_button:
            with col1:
                privacy = predict_data.T.loc[['verification_status','last_fico_range_high','last_fico_range_low','dti'],:]
                revol = predict_data.T.loc[['revol_util','mo_sin_old_rev_tl_op', 'mo_sin_rcnt_rev_tl_op'],:]
                credit = predict_data.T.loc[['open_acc','total_acc','avg_cur_bal','pct_tl_nvr_dlq'],:]
                loan = predict_data.T.loc[['term','loan_amnt','int_rate', 'sub_grade'],:]
                
                predict_prevacy = pd.DataFrame(privacy).T
                predict_prevacy.index = ['privacy']
                # predict_prevacy['last_fico_score'] = (predict_prevacy['last_fico_range_high']+predict_prevacy['last_fico_range_low']) / 2
                # del predict_prevacy['last_fico_range_high']
                # del predict_prevacy['last_fico_range_low']
                
                predict_revol = pd.DataFrame(revol).T
                predict_revol.index = ['revol']
                
                predict_credit = pd.DataFrame(credit).T
                predict_credit.index = ['credit']
                
                predict_loan = pd.DataFrame(loan).T
                predict_loan.index = ['loan']
                
                st.dataframe(predict_prevacy)
                st.dataframe(predict_revol)
                st.dataframe(predict_credit)
                st.dataframe(predict_loan)
            with col2:
                # vl = pd.DataFrame(data=[-0.0005,-0.0288,0.0003,-0.0003,0,-0.0007,-0.01,-0.008,0.0015,0.001,-0.0033,0,-0.001,0.0001,0.4692,-0.0143,])
                predict_data_preprocessed = model[:4].transform(predict_data)
                shap_value = np.array(shap.TreeExplainer(model[-1], feature_perturbation="tree_path_dependent").shap_values(predict_data_preprocessed)[0])[:, 1]
                st_shap(shap.waterfall_plot(shap.Explanation(values=shap_value, base_values=0.5, data=predict_data_preprocessed.values[0], feature_names=predict_data_preprocessed.columns.tolist())), height=500, width=900)
