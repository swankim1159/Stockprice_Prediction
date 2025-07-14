import pandas as pd
import numpy as np
import yfinance as yf
import FinanceDataReader as fdr
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# 한글 폰트 설정
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

class EnhancedARIMAPredictor:
    def __init__(self, stock_code='005930', period='5y'):
        """
        Enhanced ARIMA 모델 초기화
        
        Args:
            stock_code: 종목 코드 ('005930' for 삼성전자, '005935' for 삼성전자우)
            period: 데이터 기간 ('1y', '2y', '5y', '10y')
        """
        self.stock_code = stock_code
        self.period = period
        self.stock_data = None
        self.financial_data = None
        self.combined_data = None
        self.model = None
        self.scaler = StandardScaler()
        
    def fetch_stock_data(self):
        """주식 데이터 수집"""
        try:
            # FinanceDataReader를 사용해서 한국 주식 데이터 가져오기
            end_date = datetime.now()
            if self.period == '1y':
                start_date = end_date - timedelta(days=365)
            elif self.period == '2y':
                start_date = end_date - timedelta(days=730)
            elif self.period == '5y':
                start_date = end_date - timedelta(days=1825)
            else:  # 10y
                start_date = end_date - timedelta(days=3650)
            
            # 삼성전자 주식 데이터
            self.stock_data = fdr.DataReader(self.stock_code, start_date, end_date)
            
            # 주봉 데이터로 변환
            self.stock_data = self.stock_data.resample('W').agg({
                'Open': 'first',
                'High': 'max',
                'Low': 'min',
                'Close': 'last',
                'Volume': 'sum'
            }).dropna()
            
            print(f"✅ 주식 데이터 수집 완료: {len(self.stock_data)} 주간 데이터")
            
        except Exception as e:
            print(f"❌ 주식 데이터 수집 실패: {e}")
            # 대체 방법으로 yfinance 사용
            ticker = f"{self.stock_code}.KS"
            self.stock_data = yf.download(ticker, period=self.period, interval='1wk')
            
    def calculate_technical_indicators(self):
        """기술적 지표 계산"""
        df = self.stock_data.copy()
        
        # 이동평균선
        df['MA_5'] = df['Close'].rolling(window=5).mean()
        df['MA_20'] = df['Close'].rolling(window=20).mean()
        df['MA_60'] = df['Close'].rolling(window=60).mean()
        
        # 볼린저 밴드
        df['BB_upper'] = df['MA_20'] + (df['Close'].rolling(window=20).std() * 2)
        df['BB_lower'] = df['MA_20'] - (df['Close'].rolling(window=20).std() * 2)
        
        # RSI
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # MACD
        exp1 = df['Close'].ewm(span=12).mean()
        exp2 = df['Close'].ewm(span=26).mean()
        df['MACD'] = exp1 - exp2
        df['MACD_Signal'] = df['MACD'].ewm(span=9).mean()
        
        # 거래량 지표
        df['Volume_MA'] = df['Volume'].rolling(window=20).mean()
        df['Volume_Ratio'] = df['Volume'] / df['Volume_MA']
        
        # 가격 변동률
        df['Price_Change'] = df['Close'].pct_change()
        df['Price_Volatility'] = df['Price_Change'].rolling(window=20).std()
        
        self.stock_data = df
        
    def fetch_financial_data(self):
        """재무 지표 데이터 수집 (시뮬레이션)"""
        # 실제 환경에서는 KRX API, 네이버 증권 크롤링, 또는 유료 API 사용
        # 여기서는 시뮬레이션 데이터로 대체
        
        dates = self.stock_data.index
        np.random.seed(42)  # 재현 가능한 결과를 위해
        
        # 삼성전자 대략적인 재무 지표 범위로 시뮬레이션
        financial_indicators = {
            'Market_Cap': np.random.normal(400, 50, len(dates)),  # 시가총액 (조원)
            'PER': np.random.normal(15, 3, len(dates)),           # PER
            'PBR': np.random.normal(1.2, 0.3, len(dates)),       # PBR
            'EPS': np.random.normal(5000, 1000, len(dates)),     # EPS (원)
            'Revenue_Growth': np.random.normal(5, 10, len(dates)), # 매출 성장률 (%)
            'Operating_Margin': np.random.normal(8, 2, len(dates)), # 영업이익률 (%)
            'ROE': np.random.normal(12, 3, len(dates)),          # ROE (%)
            'Debt_Ratio': np.random.normal(15, 5, len(dates))    # 부채비율 (%)
        }
        
        self.financial_data = pd.DataFrame(financial_indicators, index=dates)
        
        # 트렌드 추가 (일부 지표에 시간에 따른 변화 반영)
        trend_factor = np.linspace(0.95, 1.05, len(dates))
        self.financial_data['Market_Cap'] *= trend_factor
        self.financial_data['EPS'] *= trend_factor
        
        print(f"✅ 재무 데이터 생성 완료: {len(self.financial_data)} 항목")
        
    def combine_data(self):
        """주식 데이터와 재무 지표 결합"""
        # 주식 데이터와 재무 데이터 결합
        self.combined_data = pd.concat([self.stock_data, self.financial_data], axis=1)
        
        # 결측치 처리
        self.combined_data = self.combined_data.fillna(method='ffill').fillna(method='bfill')
        
        # 특성 엔지니어링
        self.combined_data['Price_to_MA'] = self.combined_data['Close'] / self.combined_data['MA_20']
        self.combined_data['Volume_Price_Trend'] = (
            self.combined_data['Volume_Ratio'] * self.combined_data['Price_Change']
        )
        
        # 재무 지표 기반 특성
        self.combined_data['Valuation_Score'] = (
            self.combined_data['PER'] * 0.3 + 
            self.combined_data['PBR'] * 0.2 + 
            self.combined_data['EPS'] / 1000 * 0.5
        )
        
        print(f"✅ 데이터 결합 완료: {self.combined_data.shape}")
        
    def prepare_features(self):
        """ARIMA 모델을 위한 특성 준비"""
        # 목표 변수 (주가)
        self.target = self.combined_data['Close'].dropna()
        
        # 외생 변수 (exogenous variables) 선택
        feature_columns = [
            'Volume', 'RSI', 'MACD', 'Volume_Ratio', 'Price_Volatility',
            'PER', 'PBR', 'EPS', 'ROE', 'Market_Cap', 'Valuation_Score'
        ]
        
        self.features = self.combined_data[feature_columns].dropna()
        
        # 데이터 정규화
        self.features_scaled = pd.DataFrame(
            self.scaler.fit_transform(self.features),
            columns=self.features.columns,
            index=self.features.index
        )
        
        # 목표 변수와 특성 변수의 인덱스 맞추기
        common_index = self.target.index.intersection(self.features_scaled.index)
        self.target = self.target.loc[common_index]
        self.features_scaled = self.features_scaled.loc[common_index]
        
        print(f"✅ 특성 준비 완료: {len(self.target)} 데이터 포인트, {len(feature_columns)} 특성")
        
    def check_stationarity(self, series, title="Series"):
        """시계열 정상성 검사"""
        result = adfuller(series.dropna())
        print(f"\n=== {title} 정상성 검사 (ADF Test) ===")
        print(f"ADF Statistic: {result[0]:.6f}")
        print(f"p-value: {result[1]:.6f}")
        print(f"Critical Values:")
        for key, value in result[4].items():
            print(f"\t{key}: {value:.3f}")
        
        if result[1] <= 0.05:
            print("✅ 시계열이 정상적입니다 (Stationary)")
            return True
        else:
            print("❌ 시계열이 비정상적입니다 (Non-stationary)")
            return False
            
    def find_optimal_arima_order(self, max_p=3, max_d=2, max_q=3):
        """최적의 ARIMA 차수 찾기"""
        print("\n=== 최적 ARIMA 차수 탐색 ===")
        
        best_aic = float('inf')
        best_order = None
        results = []
        
        for p in range(max_p + 1):
            for d in range(max_d + 1):
                for q in range(max_q + 1):
                    try:
                        # 외생 변수를 포함한 ARIMAX 모델
                        model = ARIMA(
                            self.target, 
                            order=(p, d, q),
                            exog=self.features_scaled
                        )
                        fitted_model = model.fit()
                        
                        aic = fitted_model.aic
                        results.append({
                            'order': (p, d, q),
                            'aic': aic,
                            'bic': fitted_model.bic
                        })
                        
                        if aic < best_aic:
                            best_aic = aic
                            best_order = (p, d, q)
                            
                    except:
                        continue
        
        print(f"✅ 최적 차수: {best_order}, AIC: {best_aic:.2f}")
        return best_order, results
        
    def build_model(self, order=None):
        """ARIMAX 모델 구축"""
        if order is None:
            order, _ = self.find_optimal_arima_order()
        
        print(f"\n=== ARIMAX 모델 구축 ({order}) ===")
        
        # 훈련/테스트 데이터 분할
        train_size = int(len(self.target) * 0.8)
        
        self.train_target = self.target.iloc[:train_size]
        self.test_target = self.target.iloc[train_size:]
        
        self.train_features = self.features_scaled.iloc[:train_size]
        self.test_features = self.features_scaled.iloc[train_size:]
        
        # 모델 훈련
        self.model = ARIMA(
            self.train_target,
            order=order,
            exog=self.train_features
        )
        
        self.fitted_model = self.model.fit()
        
        print("✅ 모델 훈련 완료")
        print(self.fitted_model.summary())
        
    def evaluate_model(self):
        """모델 성능 평가"""
        print("\n=== 모델 성능 평가 ===")
        
        # 예측 수행
        predictions = self.fitted_model.forecast(
            steps=len(self.test_target),
            exog=self.test_features
        )
        
        # 성능 지표 계산
        mse = mean_squared_error(self.test_target, predictions)
        mae = mean_absolute_error(self.test_target, predictions)
        rmse = np.sqrt(mse)
        
        # 정확도 계산 (방향성 예측)
        actual_direction = np.diff(self.test_target) > 0
        pred_direction = np.diff(predictions) > 0
        directional_accuracy = np.mean(actual_direction == pred_direction) * 100
        
        print(f"MSE: {mse:.2f}")
        print(f"MAE: {mae:.2f}")
        print(f"RMSE: {rmse:.2f}")
        print(f"방향성 정확도: {directional_accuracy:.2f}%")
        
        # 결과 저장
        self.evaluation_results = {
            'mse': mse,
            'mae': mae,
            'rmse': rmse,
            'directional_accuracy': directional_accuracy,
            'predictions': predictions
        }
        
        return self.evaluation_results
        
    def plot_results(self):
        """결과 시각화"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. 실제 vs 예측 주가
        axes[0, 0].plot(self.test_target.index, self.test_target.values, 
                       label='Actual', color='blue', alpha=0.7)
        axes[0, 0].plot(self.test_target.index, self.evaluation_results['predictions'], 
                       label='Predicted', color='red', alpha=0.7)
        axes[0, 0].set_title('Actual vs Predicted Stock Price')
        axes[0, 0].set_xlabel('Date')
        axes[0, 0].set_ylabel('Price (KRW)')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. 잔차 분석
        residuals = self.test_target.values - self.evaluation_results['predictions']
        axes[0, 1].plot(self.test_target.index, residuals, color='green', alpha=0.7)
        axes[0, 1].axhline(y=0, color='red', linestyle='--', alpha=0.5)
        axes[0, 1].set_title('Residuals')
        axes[0, 1].set_xlabel('Date')
        axes[0, 1].set_ylabel('Residual')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. 주요 재무 지표 시각화
        axes[1, 0].plot(self.combined_data.index, self.combined_data['PER'], 
                       label='PER', alpha=0.7)
        axes[1, 0].plot(self.combined_data.index, self.combined_data['ROE'], 
                       label='ROE', alpha=0.7)
        axes[1, 0].set_title('Key Financial Indicators')
        axes[1, 0].set_xlabel('Date')
        axes[1, 0].set_ylabel('Value')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. 특성 중요도 (모델 계수)
        feature_importance = abs(self.fitted_model.params[-len(self.features_scaled.columns):])
        axes[1, 1].bar(range(len(feature_importance)), feature_importance)
        axes[1, 1].set_title('Feature Importance (|Coefficients|)')
        axes[1, 1].set_xlabel('Features')
        axes[1, 1].set_ylabel('Importance')
        axes[1, 1].set_xticks(range(len(self.features_scaled.columns)))
        axes[1, 1].set_xticklabels(self.features_scaled.columns, rotation=45)
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
    def predict_future(self, steps=4):
        """미래 주가 예측"""
        print(f"\n=== 향후 {steps}주 예측 ===")
        
        # 마지막 특성 값으로 미래 예측 (실제로는 미래 재무 지표 예측이 필요)
        future_features = pd.DataFrame(
            np.tile(self.features_scaled.iloc[-1].values, (steps, 1)),
            columns=self.features_scaled.columns
        )
        
        # 예측 수행
        future_predictions = self.fitted_model.forecast(
            steps=steps,
            exog=future_features
        )
        
        # 예측 구간 계산
        forecast_result = self.fitted_model.get_forecast(
            steps=steps,
            exog=future_features
        )
        
        conf_int = forecast_result.conf_int()
        
        # 결과 출력
        future_dates = pd.date_range(
            start=self.target.index[-1] + pd.Timedelta(weeks=1),
            periods=steps,
            freq='W'
        )
        
        for i, date in enumerate(future_dates):
            print(f"{date.strftime('%Y-%m-%d')}: "
                  f"{future_predictions.iloc[i]:,.0f}원 "
                  f"(구간: {conf_int.iloc[i, 0]:,.0f} ~ {conf_int.iloc[i, 1]:,.0f})")
        
        return future_predictions, conf_int
        
    def run_complete_analysis(self):
        """전체 분석 실행"""
        print("🚀 삼성전자 주가 예측 분석 시작")
        print("=" * 50)
        
        # 1. 데이터 수집
        self.fetch_stock_data()
        
        # 2. 기술적 지표 계산
        self.calculate_technical_indicators()
        
        # 3. 재무 데이터 생성
        self.fetch_financial_data()
        
        # 4. 데이터 결합
        self.combine_data()
        
        # 5. 특성 준비
        self.prepare_features()
        
        # 6. 정상성 검사
        self.check_stationarity(self.target, "주가")
        
        # 7. 모델 구축
        self.build_model()
        
        # 8. 모델 평가
        self.evaluate_model()
        
        # 9. 결과 시각화
        self.plot_results()
        
        # 10. 미래 예측
        future_pred, conf_int = self.predict_future()
        
        print("\n✅ 분석 완료!")
        return self.evaluation_results, future_pred, conf_int

# 사용 예시
def main():
    # 삼성전자 (005930) 분석
    print("🔍 삼성전자 (005930) 분석")
    samsung = EnhancedARIMAPredictor(stock_code='005930', period='2y')
    results_samsung, future_samsung, conf_samsung = samsung.run_complete_analysis()
    
    print("\n" + "="*60)
    
    # 삼성전자우 (005935) 분석 (선택사항)
    print("🔍 삼성전자우 (005935) 분석")
    samsung_pref = EnhancedARIMAPredictor(stock_code='005935', period='2y')
    results_pref, future_pref, conf_pref = samsung_pref.run_complete_analysis()
    
    # 두 종목 비교
    print("\n📊 종목 비교 결과")
    print("=" * 30)
    print(f"삼성전자 (005930) - 방향성 정확도: {results_samsung['directional_accuracy']:.2f}%")
    print(f"삼성전자우 (005935) - 방향성 정확도: {results_pref['directional_accuracy']:.2f}%")

if __name__ == "__main__":
    main()
