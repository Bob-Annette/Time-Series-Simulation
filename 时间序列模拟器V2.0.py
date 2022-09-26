import streamlit as st
import pandas as pd
import numpy as np
import datetime
import base64
from PIL import Image

from scipy.stats import t
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

import time
# from chinese_calendar import is_holiday
from datetime import timedelta
from typing import Any, Callable, List, Optional, Union
from pandas.tseries.frequencies import to_offset
TimedeltaLike = Union[timedelta, float, str]

st.set_page_config(
    page_title="时间序列模拟器V2.0",
    page_icon="👋",
)
def empty(*args):
    pass

class Simulator:
    def __init__(
        self,
        n: int = 100,
        freq: str = "D",
        start: Any = None,
        t0: float = 0.0,
        ts: Any = None,
    ):
        if ts is None:
            # create time
            self.n = n
            self.freq = freq
            self.start = start
            self.time = pd.date_range(
                start=start,
                freq=freq,
                periods=n,
            )
            # create the simulated time series
            self.timeseries = np.zeros(self.n) + t0
        else:
            self.time = ts.index
            self.timeseries = ts.values
            self.n = ts.shape[0]
            self.start = ts.index[0]

    def sigmoid(x: float):
        return 1 / (1 + np.exp(-10 * x))
    
    def _convert_period(self, period):
        
        return to_offset(period).nanos / 1e9
    
    def _add_component(
        self,
        component_gen: Callable,
        multiply: bool,
        time_scale: Optional[float] = None,
    ):
        timestamps = self.time.values.astype(np.float64) / 1e9
        if time_scale is None:
            time_scale = timestamps[-1] - timestamps[0] + np.finfo(float).eps
        timepoints =  np.arange(self.n) / time_scale
        component = component_gen(timepoints)

        if multiply:
            self.timeseries *= 1 + component
        else:
            self.timeseries += component

        return self
    
    # 趋势项
    def add_trend(
        self, magnitude: float, trend_type: str = "linear", multiply: bool = False
    ):
        def component_gen(timepoints):
            timestamps = self.time.values.astype(np.float64) / 1e9
            time_scale = timestamps[-1] - timestamps[0] + np.finfo(float).eps
            if trend_type == "sigmoid" or trend_type == "S":
                return magnitude * self.sigmoid(timepoints - 0.5) * time_scale
            elif trend_type == "linear" or trend_type == "L":
                return magnitude * timepoints * time_scale
            # 这里可以继续添加其它形态的时间序列
            
        return self._add_component(component_gen, multiply)
    
    # 误差项
    def add_noise(
        self,
        magnitude: float = 1.0,
        lam: float = 0.0, # 偏正态分布中的偏度参数
        multiply: bool = False,
    ):
        def component_gen(timepoints):
            return magnitude*lam/(1+lam**2)**0.5*abs(np.random.randn(len(timepoints)))+magnitude/(1+lam**2)**0.5*np.random.randn(len(timepoints))

        return self._add_component(component_gen, multiply)

    # 周期项
    def add_seasonality(
        self,
        magnitude: float = 0.0,
        period: int = 7,
        multiply: bool = False,
    ):
        def component_gen(timepoints):
            return magnitude * np.sin(np.pi * timepoints)

        return self._add_component(component_gen, multiply, time_scale=period)
    
    # 异常点
    def add_anomalies(
        self,
        anomaly_time,
        anomaly_duration: str = '1s',
        multiply: bool = True,
        magnitude: float = 1,
    ):
        anomaly_time = pd.to_datetime(anomaly_time)
        # 可以继续加其它特殊日期规则
        if multiply:
            rate = np.ones(self.n)
            for i in range(self.n):
                if self.time[i] <= anomaly_time + pd.Timedelta(anomaly_duration) and anomaly_time <= self.time[i]:
                    rate[i] = magnitude
            self.timeseries = rate * self.timeseries
        else:
            rate = np.zeros(self.n)
            for i in range(self.n):
                if self.time[i] <= anomaly_time + pd.Timedelta(anomaly_duration) and anomaly_time <= self.time[i]:
                    rate[i] = magnitude
            self.timeseries = rate + self.timeseries
        return self

    # 特殊日
    def add_special_day(
        self,
        magnitude: float = 1,
        mode: str = "special_day",
        multiply: bool = True,
        special_day = None,
    ):
        if multiply:
            rate = np.ones(self.n)
            if mode == "special_day":
                for i in range(self.n):
                    if self.time[i].year == special_day.year and self.time[i].month == special_day.month and self.time[i].day == special_day.day:
                        rate[i] = magnitude
            elif mode == "change_day":
                for i in range(self.n):
                    if self.time[i].year == special_day.year and self.time[i].month == special_day.month and self.time[i].day == special_day.day:
                        rate[i:] = magnitude
                        break
            # 可以继续加其它特殊日期规则
            self.timeseries = rate * self.timeseries
        else:
            rate = np.zeros(self.n)
            if mode == "special_day":
                for i in range(self.n):
                    if self.time[i].year == special_day.year and self.time[i].month == special_day.month and self.time[i].day == special_day.day:
                        rate[i] = magnitude
            elif mode == "change_day":
                for i in range(self.n):
                    if self.time[i].year == special_day.year and self.time[i].month == special_day.month and self.time[i].day == special_day.day:
                        rate[i:] = magnitude
                        break
            # 可以继续加其它特殊日期规则
            self.timeseries = rate + self.timeseries
        return self

    # 归一化
    def normalization(
        self,
        max_value: float,
        min_value: float,
        rate: float=1,
    ):
        ts = self.timeseries
        delta = (max_value-min_value)*rate
        if np.std(ts) == 0:
            ts[:] = (max_value-min_value)/2
            return self
        ts = ts - min(ts)
        ts = ts / (max(ts)/delta)
        ts = ts + min_value + 0.5*(max_value-min_value)*(1-rate)
        self.timeseries = ts
        return self

    # 生成模拟数
    def stl_sim(self):
        ts = pd.Series(index=self.time, data=self.timeseries, name='Value')
        return ts

def add_Special(i, start_date, end_date):
    special_days = st.sidebar.expander(f"特殊日-{i+1}")
    special_day = special_days.date_input(
        f"特殊日-{i+1}的区间", start_date, min_value=start_date, max_value=end_date
    )
    special_feature = special_days.selectbox(f"特殊日-{i+1}的特征", options=("-","特殊日内整体缩小","特殊日内整体放大"))
    return special_day, special_feature

def add_Change(i, start_date, end_date):
    change_days = st.sidebar.expander(f"变更日-{i+1}")
    change_day = change_days.date_input(
        f"变更点-{i+1}", start_date, min_value=start_date, max_value=end_date
    )
    change_feature = change_days.selectbox(f"变更点-{i+1}的特征", options=("-","变更点后整体突降","变更点后整体突增"))
    return change_day, change_feature

def add_Anomalies(i, start_date, end_date):
    Anomalies = st.sidebar.expander(f"注入异常-{i+1}")
    anomaly_start = Anomalies.date_input(
        f"异常-{i+1}开始时间", start_date, min_value=start_date, max_value=end_date
    )
    Anomalies_col1, Anomalies_col2 = Anomalies.columns(2)
    with Anomalies_col1:
        anomaly_duration = st.number_input(f"异常-{i+1}持续时长", value=1, format="%d", min_value=1)
    with Anomalies_col2:
        anomaly_unit = st.selectbox(f"异常-{i+1}单位", ('秒', '分钟', '小时', '天', '周'))
    anomaly_feature = Anomalies.selectbox(f"异常-{i+1}特征", options=("-","突降","突增"))
    anomaly_magnitude = Anomalies.slider(f"异常-{i+1}异常程度", value=0.0, format="%.1f", min_value=0.0, max_value=1.0)
    return anomaly_start, anomaly_duration, anomaly_unit, anomaly_magnitude, anomaly_feature

def GESDTAD(df):
    class GeneralizedESDTestAD():
        def __init__(self, alpha: float = 0.05) -> None:
            self.alpha = alpha

        def _fit_core(self, s: pd.Series) -> None:
            if s.count() == 0:
                raise RuntimeError("Valid values are not enough for training.")
            R = pd.Series(np.zeros(len(s)), index=s.index)
            n = s.count()
            Lambda = pd.Series(np.zeros(len(s)), index=s.index)
            s_copy = s.copy()
            i = 0
            while s_copy.count() > 0:
                i += 1
                ind = (s_copy - s_copy.mean()).abs().idxmax()
                R[ind] = (
                    abs(s_copy[ind] - s_copy.mean()) / s_copy.std()
                    if s_copy.std() > 0
                    else 0
                )
                s_copy[ind] = np.nan
                p = 1 - self.alpha / (2 * (n - i + 1))
                Lambda[ind] = (
                    (n - i)
                    * t.ppf(p, n - i - 1)
                    / np.sqrt((n - i - 1 + t.ppf(p, n - i - 1) ** 2) * (n - i + 1))
                )
                if R[ind] <= Lambda[ind]:
                    break
            self._normal_sum = s[Lambda >= R].sum()
            self._normal_squared_sum = (s[Lambda >= R] ** 2).sum()
            self._normal_count = s[Lambda >= R].count()
            i = 1
            n = self._normal_count + 1
            p = 1 - self.alpha / (2 * (n - i + 1))
            self._lambda = (
                (n - i)
                * t.ppf(p, n - i - 1)
                / np.sqrt((n - i - 1 + t.ppf(p, n - i - 1) ** 2) * (n - i + 1))
            )
            return s_copy[s_copy.isna()]

        def _predict_core(self, s: pd.Series) -> pd.Series:
            new_sum = s + self._normal_sum
            new_count = self._normal_count + 1
            new_mean = new_sum / new_count
            new_squared_sum = s ** 2 + self._normal_squared_sum
            new_std = np.sqrt(
                (
                    new_squared_sum
                    - 2 * new_mean * new_sum
                    + new_count * new_mean ** 2
                )
                / (new_count - 1)
            )
            predicted = (s - new_mean).abs() / new_std > self._lambda
            predicted[s.isna()] = np.nan
            return predicted
    if isinstance(df, pd.Series):
        df = pd.DataFrame(df.values, index=df.index, columns=['value'])
    st.subheader("广义ESD检测")
    col1, col2 = st.columns(2)
    with col1:
        alpha = st.slider('显著性水平：', min_value=0.01, max_value=0.50, value=0.05, step=0.01, format='%f')
    with col2:
        rate = st.slider('训练集比例：', min_value=0.00, max_value=1.00, value=0.30, step=0.05, format='%f')
    
    split_n = int(df.shape[0]*rate)
    ESD_ad = GeneralizedESDTestAD(alpha=alpha)
    train_anomalies = ESD_ad._fit_core(df[df.columns[0]][:split_n])
    anomalies = ESD_ad._predict_core(df[df.columns[0]][split_n:])

    fig = plt.figure(figsize=(12,5))
    train = plt.plot(df[df.columns[0]][:split_n], color='yellow', marker="o", linewidth = 3, label='Traning data', zorder=0)
    train_outliers = plt.scatter(train_anomalies.index, df.loc[train_anomalies.index, df.columns[0]].values, s=80, color='green', label='Traning outliers', zorder=2)
    test = plt.plot(df[df.columns[0]][split_n:], color='blue', marker="o", linewidth = 3, label='Test data', zorder=1)
    outliers = plt.scatter(anomalies[anomalies==True].index, df.loc[anomalies[anomalies==True].index, df.columns[0]].values, s=80, color='red', label='Test outliers', zorder=4)
    plt.title(df.columns[0], fontsize=20)
    plt.ylabel("Value", fontsize=15)
    plt.xlabel("Timestamp", fontsize=15)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.legend(fontsize=15)

    st.pyplot(fig)

def HP(df):
    def hp(ts, lam=10):
        '''
        HP滤波
        '''
        def D_matrix(N):
            '''D
            [[-1.  1.  0. ...  0.  0.  0.]
            [ 0. -1.  1. ...  0.  0.  0.]
            [ 0.  0. -1. ...  0.  0.  0.]
            ...
            [ 0.  0.  0. ...  1.  0.  0.]
            [ 0.  0.  0. ... -1.  1.  0.]
            [ 0.  0.  0. ...  0. -1.  1.]]
            '''
            D = np.zeros((N-1,N))
            D[:,1:] = np.eye(N-1)
            D[:,:-1] -= np.eye(N-1)
            return D
        N = len(ts)
        D = D_matrix(N-1).dot(D_matrix(N))
        g = np.linalg.inv((np.eye(N)+lam*D.T.dot(D))).dot(ts)
        return g
    # if isinstance(df, pd.Series):
    #     df = pd.DataFrame(df.values, index=df.index, columns=['value'])
    lam = st.slider("lambda系数", value=0.0, format="%f", min_value=0.0, max_value=10.0)
    hp_df = pd.Series(hp(df.values, lam=lam), index=df.index)
    fig = plt.figure(figsize=(12,5))
    plt.plot(df, label='original')
    plt.plot(hp_df, label='filtered')
    st.pyplot(fig)


def intro():
    st.title('时间序列模拟器V2.0')
    image = Image.open('背景.png')
    st.image(image, caption='')

    with st.expander("💬评论区"):
        comment_df = pd.read_excel("评论.xlsx")
        for i in range(comment_df.shape[0]):
            st.markdown(f">{comment_df.loc[i,'姓名']}—{comment_df.loc[i,'时间']}")
            st.markdown(f"{comment_df.loc[i,'内容']}")

        st.write("**添加你的评论**")
        form = st.form("comment")
        name = form.text_input("姓名", value="")
        comment = form.text_area("内容", value="")
        submit = form.form_submit_button("添加")

        if submit:
            comment_df = comment_df.append(pd.DataFrame({"时间":[pd.to_datetime(datetime.datetime.now()).strftime("%Y/%m/%d %H:%M:%S")], "姓名":[name], "内容":[comment]}), ignore_index=True)
            comment_df.to_excel("评论.xlsx", index=False)
            st.experimental_rerun()

def get_table_download_link(df):
    csv = df.to_csv(index=True)
    b64 = base64.b64encode(
        csv.encode()
    ).decode()  # some strings <-> bytes conversions necessary here
    href = f'<a href="data:file/csv;base64,{b64}" download="时间序列模拟.csv">下载数据</a>'
    return href

def custom_demo():
    st.header("时间序列模拟")
    # 时间序列的时间戳
    st.subheader("基础属性") # 添加副标题
    col1, col2, col3 = st.columns(3)
    with col1:
        start_date = st.date_input(
            "起始时间", datetime.date(2022, 9, 1), min_value=datetime.date(2020, 1, 1), max_value=datetime.date(2022, 12, 31)
        )
    with col2:
        granularity_value = st.number_input(
            "颗粒度", value=1, format="%d", min_value=1
        )
    with col3:
        granularity = st.selectbox(
            "颗粒度单位", ('秒', '分钟', '小时', '天', '周'), index=2
        )
    granularity_dir = {'秒':'s', '分钟':'min', '小时':'h', '天':'d', '周':'w'}
    col1, col2, col3 = st.columns(3)
    with col1:
        end_date = st.date_input(
            "结束时间", datetime.date(2022, 9, 30), min_value=start_date, max_value=datetime.date(2022, 12, 31)
        )
    with col2:
        max_value = st.number_input(
            "最大值", value=100, format="%d"
        )
    with col3:
        min_value = st.number_input(
            "最小值", value=0, format="%d"
        )

    # 画图
    if granularity == '秒':
        n = int((end_date-start_date)/datetime.timedelta(seconds=granularity_value)) # 样本个数
    elif granularity == '分钟':
        n = int((end_date-start_date)/datetime.timedelta(minutes=granularity_value))
    elif granularity == '小时':
        n = int((end_date-start_date)/datetime.timedelta(hours=granularity_value))
    elif granularity == '天':
        n = int((end_date-start_date)/datetime.timedelta(days=granularity_value))
    elif granularity == '周':
        n = int((end_date-start_date)/datetime.timedelta(weeks=granularity_value))
    
    # 初始模拟器
    sim = Simulator(n=n, start=start_date, freq=''.join([str(int(granularity_value)), granularity_dir[granularity]]), t0=min_value)
    
    # 高级属性
    st.sidebar.header("高级属性")
    box_features = st.sidebar.expander("开箱特征")
    day_classics = box_features.checkbox("经典天周期")
    if day_classics:
        sim.add_seasonality(magnitude=10, period=int(n/((sim.time[-1] - sim.time[0]).value/1e9/86400)/2))
    week_classics = box_features.checkbox("经典周周期")
    if week_classics:
        sim.add_seasonality(magnitude=10, period=int(n/((sim.time[-1] - sim.time[0]).value/1e9/604800)/2))
    clear_regularly = box_features.checkbox("定时清理")
    run_regularly = box_features.checkbox("定时跑批")
    if run_regularly:
        for i in pd.date_range(start=start_date,freq='d',end=end_date):
            sim.add_anomalies(anomaly_time=i,multiply=False,magnitude=0.5*(max_value-min_value))

    custom_features = st.sidebar.expander("自定义特征")
    Trend = custom_features.checkbox("趋势")
    if Trend:
        Trend_col1, Trend_col2 = custom_features.columns(2)
        with Trend_col1:
            trend_mode = st.selectbox("趋势", options=("上升","下降"), disabled=not Trend)
        with Trend_col2:
            magnitude_trend = st.slider("倾斜程度", value=0.00, format="%.2f", min_value=0.00, max_value=1.00, step=0.05, disabled=not Trend)
        magnitude_trend = (max_value-min_value)/n*magnitude_trend
        if trend_mode == "下降":
            sim.timeseries = sim.timeseries - min_value + max_value
            magnitude_trend = -1*magnitude_trend
        # 添加趋势项
        sim.add_trend(magnitude=magnitude_trend)
    
    Seasonal = custom_features.checkbox("周期")
    if Seasonal:
        magnitude_seasonality = 10 # 振幅，先锁住
        Seasonal_col1, Seasonal_col2 = custom_features.columns(2)
        with Seasonal_col1:
            periods_value = st.number_input("周期值", value=1, format="%d", min_value=1, disabled=not Seasonal)
        with Seasonal_col2:
            periods = st.selectbox("周期单位", ('秒', '分钟', '小时', '天', '周'), index=3, disabled=not Seasonal)
        period_dic = {'s':1, 'min':60, 'h':3600, 'd':86400, 'w':604800}
        period = period_dic[granularity_dir[periods]]*periods_value
        period = int(n/((sim.time[-1] - sim.time[0]).value/1e9/period)/2)
        sim.add_seasonality(magnitude=magnitude_seasonality, period=period)

    Noise = custom_features.checkbox("残差")
    if Noise:
        Noise_col1, Noise_col2 = custom_features.columns(2)
        with Noise_col1:
            magnitude_noise = st.slider("波动程度", value=1.00, format="%.2f", min_value=0.00, max_value=10.00, disabled=not Noise)
        with Noise_col2:
            lam = st.slider("偏度", value=0.0, format="%.1f", min_value=-10.0, max_value=10.0, disabled=not Noise)
        sim.add_noise(magnitude=magnitude_noise, lam=lam)
    
    # 定时清理
    if clear_regularly:
        for i in pd.date_range(start=start_date+pd.Timedelta('1d'),freq='d',end=end_date):
            ts = sim.stl_sim()
            sim.add_special_day(magnitude=-0.4*(max(ts[i - pd.Timedelta('1d'):i])-min(ts[i - pd.Timedelta('1d'):i])), mode="change_day", multiply=False, special_day=i)

    st.sidebar.header("特殊日/变更日")
    Specials_nums = st.sidebar.number_input("特殊日个数", value=0, format="%d", min_value=0)
    Specials_dir = {}
    for i in range(Specials_nums):
        Specials_dir[f"specials_{i}"] = add_Special
        special_day, special_feature = Specials_dir[f"specials_{i}"](i, start_date, end_date)
        if special_feature == "特殊日内整体缩小":
            sim.add_special_day(magnitude=0.6, special_day=special_day)
        elif special_feature == "特殊日内整体放大":
            sim.add_special_day(magnitude=1.4, special_day=special_day)

    Change_nums = st.sidebar.number_input("变更日个数", value=0, format="%d", min_value=0)
    Change_dir = {}
    for i in range(Change_nums):
        Change_dir[f"change_{i}"] = add_Change
        change_day, change_feature = Change_dir[f"change_{i}"](i, start_date, end_date)
        if change_feature == "变更点后整体突降":
            sim.add_special_day(magnitude=0.6, special_day=change_day, mode="change_day")
        elif change_feature == "变更点后整体突增":
            sim.add_special_day(magnitude=1.4, special_day=change_day, mode="change_day")

    st.sidebar.header("异常注入")
    Anomalies_nums = st.sidebar.number_input("异常个数", value=0, format="%d", min_value=0)
    Anomalies_dir = {}
    for i in range(Anomalies_nums):
        Anomalies_dir[f"anomalies_{i}"] = add_Anomalies
        anomaly_start, anomaly_duration, anomaly_unit, anomaly_magnitude, anomaly_feature = Anomalies_dir[f"anomalies_{i}"](i, start_date, end_date)
        if anomaly_feature == "突增":
            sim.add_anomalies(anomaly_time=anomaly_start,anomaly_duration=f'{anomaly_duration}{granularity_dir[anomaly_unit]}',magnitude=np.pi**anomaly_magnitude)
        elif anomaly_feature == "突降":
            sim.add_anomalies(anomaly_time=anomaly_start,anomaly_duration=f'{anomaly_duration}{granularity_dir[anomaly_unit]}',magnitude=0.3**anomaly_magnitude)
    
    # 生成时间序列
    if min(sim.timeseries)<min_value or max(sim.timeseries)>max_value:
        sim.normalization(max_value=max_value,min_value=min_value,rate=0.9)
    ts = sim.stl_sim()
    st.line_chart(ts)

    # 展示和下载数据
    col1, col2, col3 = st.columns(3)
    with col1:
        show_base_df = st.checkbox("详细数据")
    with col2:
        topn = st.number_input("展示前N行", value=20, min_value=1, format="%d", disabled=not show_base_df)
    with col3:
        st.markdown(get_table_download_link(ts), unsafe_allow_html=True) # 下载数据
    if show_base_df:
        st.dataframe(ts.head(topn))
    
    st.header("异常检测")
    detect_model = st.selectbox("检测模型", AD_types.keys())
    AD_types[detect_model](ts)

def import_demo():
    granularity_dir = {'秒':'s', '分钟':'min', '小时':'h', '天':'d', '周':'w'}
    uploaded_file = st.file_uploader("导入外部数据") # 上传文件
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        # 设置时间戳
        try:
            df[df.columns[0]] = df[df.columns[0]].apply(lambda t:str(time.strftime("%Y-%m-%d %H:%M:%S",time.localtime(int(str(t)[:10])))))
        except:
            pass
        df[df.columns[0]] = pd.to_datetime(df[df.columns[0]])
        df = df.sort_values(by=[df.columns[0]],ascending=True)
        df = df.set_index(df.columns[0])
        ts = df[df.columns[0]]
        sim = Simulator(ts=ts)
        start_date = sim.time[0]
        end_date = sim.time[-1]

        st.sidebar.header("特殊日/变更日")
        Specials_nums = st.sidebar.number_input("特殊日个数", value=0, format="%d", min_value=0)
        Specials_dir = {}
        for i in range(Specials_nums):
            Specials_dir[f"specials_{i}"] = add_Special
            special_day, special_feature = Specials_dir[f"specials_{i}"](i, start_date, end_date)
            if special_feature == "特殊日内整体缩小":
                sim.add_special_day(magnitude=0.6, special_day=special_day)
            elif special_feature == "特殊日内整体放大":
                sim.add_special_day(magnitude=1.4, special_day=special_day)

        Change_nums = st.sidebar.number_input("变更日个数", value=0, format="%d", min_value=0)
        Change_dir = {}
        for i in range(Change_nums):
            Change_dir[f"change_{i}"] = add_Change
            change_day, change_feature = Change_dir[f"change_{i}"](i, start_date, end_date)
            if change_feature == "变更点后整体突降":
                sim.add_special_day(magnitude=0.6, special_day=change_day, mode="change_day")
            elif change_feature == "变更点后整体突增":
                sim.add_special_day(magnitude=1.4, special_day=change_day, mode="change_day")

        st.sidebar.header("异常注入")
        Anomalies_nums = st.sidebar.number_input("异常个数", value=0, format="%d", min_value=0)
        Anomalies_dir = {}
        for i in range(Anomalies_nums):
            Anomalies_dir[f"anomalies_{i}"] = add_Anomalies
            anomaly_start, anomaly_duration, anomaly_unit, anomaly_magnitude, anomaly_feature = Anomalies_dir[f"anomalies_{i}"](i, start_date, end_date)
            if anomaly_feature == "突增":
                sim.add_anomalies(anomaly_time=anomaly_start,anomaly_duration=f'{anomaly_duration}{granularity_dir[anomaly_unit]}',magnitude=np.pi**anomaly_magnitude)
            elif anomaly_feature == "突降":
                sim.add_anomalies(anomaly_time=anomaly_start,anomaly_duration=f'{anomaly_duration}{granularity_dir[anomaly_unit]}',magnitude=0.3**anomaly_magnitude)

        ts = sim.stl_sim()
        st.line_chart(ts)

        # 展示和下载数据
        col1, col2, col3 = st.columns(3)
        with col1:
            show_base_df = st.checkbox("详细数据")
        with col2:
            topn = st.number_input("展示前N行", value=20, min_value=1, format="%d", disabled=not show_base_df)
        with col3:
            st.markdown(get_table_download_link(ts), unsafe_allow_html=True) # 下载数据
        if show_base_df:
            st.dataframe(ts.head(topn))
        
        st.header("异常检测")
        detect_model = st.selectbox("检测模型", AD_types.keys())
        AD_types[detect_model](ts)

def test_demo():
    # simpson积分
    def simpson(func, start, end, n):
        x = np.linspace(start, end, 2*n+1)
        res = 0
        res = res+func(x[0])+func(x[-1])
        for i in range(n):
            res = res + 4*func(x[2*i+1])
        for i in range(n-1):
            res = res+ 2*func(x[2*i+2])
        return res*(end-start)/(6*n)

    # 傅里叶级数展开
    def Fourier(
        func: Callable,
        t0: float,
        T: float,
        n: int = 10,
    ):
        def function(x):
            a0 = 2/T*simpson(func=func, start=t0, end=t0+T, n=50)
            f = a0/2
            for i in range(1,n+1):
                def funccos(x):
                    return func(x)*np.cos(i*x)
                def funcsin(x):
                    return func(x)*np.sin(i*x)
                an = 2/T*simpson(func=funccos, start=t0, end=t0+T, n=50)
                bn = 2/T*simpson(func=funcsin, start=t0, end=t0+T, n=50)
                f = f + an*np.cos(i*x) + bn*np.sin(i*x)
            return f
        return function

    def constant_func(x): # 常值函数
        return 3

    def positive_func(x): # 正比例函数
        return x

    func_dir = {
        # "y=c": constant_func,
        "y=x": positive_func,
    }

    st.header("功能测试——傅里叶级数周期")
    st.subheader("傅里叶级数周期")
    st.markdown(
        r'''
        周期函数表达式：
        $$
        f(x)=f(x+kT) \quad (k=1,2,3,\dots)
        $$
        如果该周期函数满足狄利赫里条件，那么该周期可以展开为傅里叶级数：
        $$
        f(t)=\frac{a_0}{2}+\sum_{n=1}^{\infty} \left[ a_n \cos (n \omega_1 t)+b_n \sin (n \omega_1 t) \right]
        $$
        其中傅里叶系数计算如下：
        $$
        \begin{align*}
        \frac{a_0}{2}=&\frac{1}{T} \int_{t_0}^{t_0+T}f(t)dt \\
        a_n=&\frac{2}{T} \int_{t_0}^{t_0+T}f(t)\cos (n \omega_1 t)dt \\
        b_n=&\frac{2}{T} \int_{t_0}^{t_0+T}f(t)\sin (n \omega_1 t)dt \\
        \end{align*}
        $$
        '''
    )
    st.markdown(
        r'''
        狄利赫里条件：对于周期长度为$2L$的周期函数$f(x)$，必须满足以下三个条件

        1、在周期$2L$内，函数$f(x)$连续或只有有限个第一类间断点；

        2、在周期$2L$内，函数$f(x)$的极大值和极小值的数目应是有限个；

        3、在周期$2L$内，函数$f(x)$是绝对可积的。
        '''
    )

    col1, col2, col3 = st.columns(3)
    t0 = 0
    with col1:
        func = st.selectbox("选择基础函数", func_dir.keys())
    with col2:
        T = st.number_input("设置周期", value=3, min_value=1, format="%d")
    with col3:
        n = st.number_input("设置级数", value=10, min_value=1, format="%d")

    x = np.linspace(0,10,100)
    f = Fourier(func=func_dir[func], t0=t0, T=T, n=n)
    y = f(x)

    fig = plt.figure(figsize=(12,5))
    plt.plot(x,y)
    st.pyplot(fig)


simulation_types = {
    "—": intro,
    "自定义模拟": custom_demo,
    "导入外部数据": import_demo,
    "功能测试": test_demo,
}

AD_types = {
    "—": empty,
    "广义ESD检测": GESDTAD,
    "HP滤波": HP,
}

demo_name = st.sidebar.selectbox("选择模拟方式", simulation_types.keys())
simulation_types[demo_name]()