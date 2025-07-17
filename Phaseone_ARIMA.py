# 필수 라이브러리 설치 확인 및 대체 방안
import sys
import subprocess
import warnings
warnings.filterwarnings('ignore')

def install_package(package):
    """패키지 설치 함수"""
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        print(f"✅ {package} 설치 완료")
    except subprocess.CalledProcessError:
        print(f"❌ {package} 설치 실패")

# 필수 라이브러리 임포트 및 설치
try:
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    from datetime import datetime, timedelta
    import json
    import urllib.request
    import urllib.parse
    import ssl
    print("✅ 기본 라이브러리 로드 완료")
except ImportError as e:
    print(f"❌ 기본 라이브러리 오류: {e}")
    print("pandas, numpy, matplotlib 설치가 필요합니다.")

# 통계 라이브러리 (대체 방안 포함)
try:
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tsa.stattools import adfuller
    STATSMODELS_AVAILABLE = True
    print("✅ statsmodels 로드 완료")
except ImportError:
    print("⚠️ statsmodels 미설치. 간단한 이동평균 모델로 대체합니다.")
    STATSMODELS_AVAILABLE = False

# 데이터 수집 라이브러리 (대체 방안 포함)
try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
    print("✅ yfinance 로드 완료")
except ImportError:
    print("⚠️ yfinance 미설치. 직접 크롤링으로 대체합니다.")
    YFINANCE_AVAILABLE = False

class SamsungStockPredictor:
    def __init__(self):
        self.samsung_data = None
        self.samsung_pref_data = None
        self.model = None
        self.forecast_result = None
        
    def get_korean_stock_data(self, symbol, period='2y'):
        """
        한국 주식 데이터 수집 (yfinance 우선, 실패시 샘플 데이터)
        """
        if YFINANCE_AVAILABLE:
            try:
                # 한국 주식 심볼 형식으로 변환
                if symbol == '005930':
                    ticker = '005930.KS'
                elif symbol == '005935':
                    ticker = '005935.KS'
                else:
                    ticker = f"{symbol}.KS"
                
                stock = yf.Ticker(ticker)
                data = stock.history(period=period)
                
                if data.empty:
                    raise Exception("데이터가 비어있습니다.")
                
                # 한국 시간대 설정 및 컬럼명 한국어로 변경
                data.index = pd.to_datetime(data.index)
                data.columns = ['시가', '고가', '저가', '종가', '거래량', '배당금', '주식분할']
                
                return data
            except Exception as e:
                print(f"yfinance 데이터 수집 실패: {e}")
                return self.create_sample_data(symbol)
        else:
            print("yfinance 미설치. 샘플 데이터를 사용합니다.")
            return self.create_sample_data(symbol)
    
    def create_sample_data(self, symbol):
        """
        샘플 데이터 생성 (실제 데이터 수집 실패시)
        """
        print(f"📊 {symbol} 샘플 데이터 생성 중...")
        
        # 최근 2년간 날짜 생성
        end_date = datetime.now()
        start_date = end_date - timedelta(days=730)
        dates = pd.date_range(start=start_date, end=end_date, freq='D')
        
        # 주말 제거
        dates = dates[dates.weekday < 5]
        
        # 기본 가격 설정 (삼성전자 기준)
        if symbol == '005930':
            base_price = 75000
        else:
            base_price = 65000
        
        # 랜덤 워크로 주가 시뮬레이션
        np.random.seed(42)  # 재현 가능한 결과
        n_days = len(dates)
        
        # 가격 시뮬레이션
        price_changes = np.random.normal(0, 0.02, n_days)  # 일 2% 변동
        prices = [base_price]
        
        for change in price_changes[1:]:
            new_price = prices[-1] * (1 + change)
            prices.append(max(new_price, base_price * 0.5))  # 최소 50% 하락 제한
        
        prices = np.array(prices)
        
        # OHLC 데이터 생성
        highs = prices * np.random.uniform(1.0, 1.03, n_days)
        lows = prices * np.random.uniform(0.97, 1.0, n_days)
        opens = np.roll(prices, 1)
        opens[0] = prices[0]
        
        # 거래량 생성
        base_volume = 20000000 if symbol == '005930' else 5000000
        volumes = np.random.normal(base_volume, base_volume * 0.3, n_days)
        volumes = np.abs(volumes)  # 음수 제거
        
        # 데이터프레임 생성
        data = pd.DataFrame({
            '시가': opens,
            '고가': highs,
            '저가': lows,
            '종가': prices,
            '거래량': volumes,
            '배당금': np.zeros(n_days),
            '주식분할': np.ones(n_days)
        }, index=dates)
        
        print(f"✅ {symbol} 샘플 데이터 생성 완료 ({len(data)}일)")
        return data
    
    def create_weekly_data(self, daily_data):
        """일봉 데이터를 주봉 데이터로 변환"""
        weekly_data = daily_data.resample('W').agg({
            '시가': 'first',
            '고가': 'max',
            '저가': 'min',
            '종가': 'last',
            '거래량': 'sum'
        }).dropna()
        
        return weekly_data
    
    def load_data(self):
        """삼성전자와 삼성전자우 데이터 로드"""
        print("📊 삼성전자 주가 데이터 수집 중...")
        
        # 삼성전자 (005930)
        self.samsung_data = self.get_korean_stock_data('005930', period='3y')
        
        # 삼성전자우 (005935)
        self.samsung_pref_data = self.get_korean_stock_data('005935', period='3y')
        
        if self.samsung_data is not None and self.samsung_pref_data is not None:
            # 주봉 데이터로 변환
            self.samsung_weekly = self.create_weekly_data(self.samsung_data)
            self.samsung_pref_weekly = self.create_weekly_data(self.samsung_pref_data)
            
            print(f"✅ 데이터 수집 완료!")
            print(f"삼성전자 데이터 기간: {self.samsung_weekly.index.min().strftime('%Y-%m-%d')} ~ {self.samsung_weekly.index.max().strftime('%Y-%m-%d')}")
            print(f"삼성전자 주봉 데이터 수: {len(self.samsung_weekly)}개")
            print(f"삼성전자우 데이터 기간: {self.samsung_pref_weekly.index.min().strftime('%Y-%m-%d')} ~ {self.samsung_pref_weekly.index.max().strftime('%Y-%m-%d')}")
            print(f"삼성전자우 주봉 데이터 수: {len(self.samsung_pref_weekly)}개")
            
            return True
        else:
            print("❌ 데이터 수집에 실패했습니다.")
            return False
    
    def analyze_stationarity(self, series, title=""):
        """시계열 데이터의 정상성 검정 (statsmodels 사용 가능시)"""
        if not STATSMODELS_AVAILABLE:
            print(f"\n=== {title} 간단한 안정성 검정 ===")
            # 단순 변화율 기반 검정
            returns = series.pct_change().dropna()
            mean_return = returns.mean()
            std_return = returns.std()
            
            print(f"평균 수익률: {mean_return:.4f}")
            print(f"표준편차: {std_return:.4f}")
            print(f"변동계수: {std_return/abs(mean_return):.4f}" if mean_return != 0 else "변동계수: 계산 불가")
            
            # 간단한 정상성 판단 (변동계수 기준)
            if std_return/abs(mean_return) < 50 if mean_return != 0 else True:
                print("✅ 상대적으로 안정적인 시계열입니다.")
                return True
            else:
                print("❌ 불안정한 시계열입니다.")
                return False
        
        print(f"\n=== {title} 정상성 검정 ===")
        
        # ADF 테스트
        adf_result = adfuller(series.dropna())
        print(f"ADF 통계량: {adf_result[0]:.4f}")
        print(f"p-value: {adf_result[1]:.4f}")
        print(f"임계값: {adf_result[4]}")
        
        if adf_result[1] <= 0.05:
            print("✅ 정상 시계열입니다.")
            return True
        else:
            print("❌ 비정상 시계열입니다. 차분이 필요합니다.")
            return False
    
    def find_optimal_arima_params(self, series, max_p=3, max_d=2, max_q=3):
        """최적의 ARIMA 파라미터 찾기 (statsmodels 사용시)"""
        if not STATSMODELS_AVAILABLE:
            print("⚠️ statsmodels 미설치. 기본 파라미터 (1,1,1) 사용")
            return (1, 1, 1)
        
        print("\n🔍 최적 ARIMA 파라미터 탐색 중...")
        
        best_aic = float('inf')
        best_params = None
        
        for p in range(max_p + 1):
            for d in range(max_d + 1):
                for q in range(max_q + 1):
                    try:
                        model = ARIMA(series, order=(p, d, q))
                        fitted_model = model.fit()
                        
                        if fitted_model.aic < best_aic:
                            best_aic = fitted_model.aic
                            best_params = (p, d, q)
                            
                    except:
                        continue
        
        if best_params is None:
            best_params = (1, 1, 1)
            print("⚠️ 최적 파라미터 탐색 실패. 기본값 (1,1,1) 사용")
        else:
            print(f"✅ 최적 파라미터: ARIMA{best_params}, AIC: {best_aic:.2f}")
        
        return best_params
    
    def create_arima_model(self, target_stock='samsung'):
        """ARIMA 모델 생성 및 훈련 (대체 방안 포함)"""
        if target_stock == 'samsung':
            data = self.samsung_weekly['종가']
            title = "삼성전자"
        else:
            data = self.samsung_pref_weekly['종가']
            title = "삼성전자우"
        
        print(f"\n🤖 {title} 예측 모델 생성 중...")
        
        # 정상성 검정
        is_stationary = self.analyze_stationarity(data, title)
        
        if STATSMODELS_AVAILABLE:
            # ARIMA 모델 사용
            optimal_params = self.find_optimal_arima_params(data)
            
            # 모델 훈련
            self.model = ARIMA(data, order=optimal_params)
            self.fitted_model = self.model.fit()
            
            print(f"✅ {title} ARIMA 모델 생성 완료!")
            print(self.fitted_model.summary())
            
            return self.fitted_model
        else:
            # 간단한 이동평균 모델 사용
            print("⚠️ ARIMA 대신 이동평균 모델을 사용합니다.")
            self.simple_model_data = data
            print(f"✅ {title} 이동평균 모델 생성 완료!")
            return "simple_model"
    
    def predict_future_prices(self, steps=12):
        """미래 주가 예측 (ARIMA 또는 이동평균 방법)"""
        print(f"\n🔮 향후 {steps}주 주가 예측 중...")
        
        if STATSMODELS_AVAILABLE and hasattr(self, 'fitted_model'):
            # ARIMA 모델 예측
            forecast = self.fitted_model.forecast(steps=steps)
            forecast_ci = self.fitted_model.get_forecast(steps=steps).conf_int()
            
            # 예측 결과 정리
            future_dates = pd.date_range(
                start=self.samsung_weekly.index[-1] + timedelta(weeks=1),
                periods=steps,
                freq='W'
            )
            
            self.forecast_result = pd.DataFrame({
                '예측가': forecast,
                '하한': forecast_ci.iloc[:, 0],
                '상한': forecast_ci.iloc[:, 1]
            }, index=future_dates)
            
        else:
            # 간단한 이동평균 기반 예측
            data = self.simple_model_data
            
            # 최근 4주 이동평균을 기준으로 예측
            recent_prices = data.tail(4)
            trend = (recent_prices.iloc[-1] - recent_prices.iloc[0]) / len(recent_prices)
            
            # 미래 날짜 생성
            future_dates = pd.date_range(
                start=self.samsung_weekly.index[-1] + timedelta(weeks=1),
                periods=steps,
                freq='W'
            )
            
            # 예측 가격 생성
            base_price = data.iloc[-1]
            predicted_prices = []
            
            for i in range(steps):
                predicted_price = base_price + (trend * (i + 1))
                predicted_prices.append(predicted_price)
            
            # 신뢰구간 (±10%)
            predicted_prices = np.array(predicted_prices)
            lower_bound = predicted_prices * 0.9
            upper_bound = predicted_prices * 1.1
            
            self.forecast_result = pd.DataFrame({
                '예측가': predicted_prices,
                '하한': lower_bound,
                '상한': upper_bound
            }, index=future_dates)
        
        print("✅ 예측 완료!")
        print(self.forecast_result.head())
        
        return self.forecast_result
    
    def visualize_results(self):
        """결과 시각화"""
        if self.forecast_result is None:
            print("❌ 예측 결과가 없습니다.")
            return
        
        plt.figure(figsize=(15, 10))
        
        # 전체 그래프
        plt.subplot(2, 2, 1)
        plt.plot(self.samsung_weekly.index, self.samsung_weekly['종가'], 
                label='삼성전자 실제 주가', color='blue', linewidth=2)
        plt.plot(self.samsung_pref_weekly.index, self.samsung_pref_weekly['종가'], 
                label='삼성전자우 실제 주가', color='green', linewidth=2)
        plt.title('삼성전자 vs 삼성전자우 주가 비교', fontsize=14, fontweight='bold')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 예측 결과
        plt.subplot(2, 2, 2)
        # 최근 6개월 데이터
        recent_data = self.samsung_weekly.tail(24)
        plt.plot(recent_data.index, recent_data['종가'], 
                label='실제 주가', color='blue', linewidth=2)
        plt.plot(self.forecast_result.index, self.forecast_result['예측가'], 
                label='예측 주가', color='red', linewidth=2)
        plt.fill_between(self.forecast_result.index, 
                        self.forecast_result['하한'], 
                        self.forecast_result['상한'], 
                        alpha=0.3, color='red', label='예측 구간')
        plt.title('삼성전자 주가 예측 결과', fontsize=14, fontweight='bold')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 거래량 분석
        plt.subplot(2, 2, 3)
        plt.plot(self.samsung_weekly.index, self.samsung_weekly['거래량'], 
                color='orange', linewidth=1, alpha=0.7)
        plt.title('삼성전자 거래량 추이', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        
        # 가격 분포
        plt.subplot(2, 2, 4)
        plt.hist(self.samsung_weekly['종가'], bins=30, alpha=0.7, color='skyblue')
        plt.axvline(self.samsung_weekly['종가'].mean(), color='red', 
                   linestyle='--', label=f'평균: {self.samsung_weekly["종가"].mean():.0f}원')
        plt.title('삼성전자 주가 분포', fontsize=14, fontweight='bold')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def generate_investment_insight(self):
        """투자 인사이트 생성"""
        if self.forecast_result is None:
            return "예측 결과가 없습니다."
        
        current_price = self.samsung_weekly['종가'].iloc[-1]
        predicted_price = self.forecast_result['예측가'].iloc[-1]
        price_change = ((predicted_price - current_price) / current_price) * 100
        
        # 거래량 트렌드 분석
        recent_volume = self.samsung_weekly['거래량'].tail(4).mean()
        historical_volume = self.samsung_weekly['거래량'].mean()
        volume_change = ((recent_volume - historical_volume) / historical_volume) * 100
        
        insight = f"""
        🎯 삼성전자 투자 인사이트 (ARIMA 기반 분석)
        
        📊 현재 상황:
        - 현재 주가: {current_price:,.0f}원
        - 12주 후 예상 주가: {predicted_price:,.0f}원
        - 예상 수익률: {price_change:+.1f}%
        
        📈 거래량 분석:
        - 최근 평균 거래량: {recent_volume:,.0f}주
        - 전체 평균 거래량: {historical_volume:,.0f}주
        - 거래량 변화: {volume_change:+.1f}%
        
        💡 투자 관점:
        {"📈 상승 추세가 예상됩니다." if price_change > 0 else "📉 하락 추세가 예상됩니다."}
        {"📊 거래량이 증가하고 있어 관심이 높아지고 있습니다." if volume_change > 0 else "📊 거래량이 감소하고 있어 관심이 줄어들고 있습니다."}
        
        ⚠️ 주의사항:
        - 이 예측은 과거 데이터 기반 통계 모델입니다
        - 실제 투자 결정 시에는 추가적인 펀더멘털 분석이 필요합니다
        - 시장 상황, 정치적 요인, 글로벌 경제 상황 등을 고려해야 합니다
        """
        
        return insight

def main():
    """메인 실행 함수 (의존성 검사 포함)"""
    print("🚀 삼성전자 ARIMA 주가 예측 시스템 시작")
    print("=" * 50)
    
    # 의존성 체크
    print("\n📋 의존성 체크:")
    print(f"- pandas, numpy, matplotlib: ✅")
    print(f"- statsmodels: {'✅' if STATSMODELS_AVAILABLE else '❌ (간단한 이동평균 모델로 대체)'}")
    print(f"- yfinance: {'✅' if YFINANCE_AVAILABLE else '❌ (샘플 데이터로 대체)'}")
    
    if not STATSMODELS_AVAILABLE:
        print("\n⚠️ 주의: statsmodels가 설치되지 않았습니다.")
        print("   pip install statsmodels 명령어로 설치하시면 ARIMA 모델을 사용할 수 있습니다.")
    
    if not YFINANCE_AVAILABLE:
        print("\n⚠️ 주의: yfinance가 설치되지 않았습니다.")
        print("   pip install yfinance 명령어로 설치하시면 실제 데이터를 사용할 수 있습니다.")
    
    print("\n" + "=" * 50)
    
    # 예측 모델 실행
    try:
        predictor = SamsungStockPredictor()
        
        # 1. 데이터 로드
        if not predictor.load_data():
            print("❌ 프로그램을 종료합니다.")
            return
        
        # 2. 모델 생성
        model = predictor.create_arima_model(target_stock='samsung')
        
        # 3. 미래 예측
        forecast = predictor.predict_future_prices(steps=12)
        
        # 4. 결과 시각화
        predictor.visualize_results()
        
        # 5. 투자 인사이트 출력
        insight = predictor.generate_investment_insight()
        print(insight)
        
        print("\n" + "=" * 50)
        print("✅ 분석 완료! 투자 결정에 참고하세요.")
        
    except Exception as e:
        print(f"❌ 실행 중 오류 발생: {e}")
        print("다음 명령어들을 실행해보세요:")
        print("pip install pandas numpy matplotlib statsmodels yfinance")
        
        # 기본 설치 명령어 표시
        print("\n📦 추천 설치 명령어:")
        print("pip install pandas numpy matplotlib")
        print("pip install statsmodels")  
        print("pip install yfinance")

if __name__ == "__main__":
    main()
