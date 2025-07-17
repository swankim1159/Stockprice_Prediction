import pandas as pd
import numpy as np
import yfinance as yf
import FinanceDataReader as fdr
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import requests
import json
import time
import schedule
from datetime import datetime, timedelta
import warnings
import logging
from pathlib import Path
import pickle
import sqlite3
warnings.filterwarnings('ignore')

class IntegratedStockAlertSystem:
    def __init__(self, config_path='config.json'):
        """
        통합 주식 알림 시스템 초기화
        
        Args:
            config_path: 설정 파일 경로
        """
        self.config = self.load_config(config_path)
        self.setup_logging()
        self.setup_database()
        
        # 종목 설정 (종목 코드 수정)
        self.symbols = {
            'samsung': '005930',  # 삼성전자 (수정됨)
            'samsung_pref': '005935'  # 삼성전자우
        }
        
        # ARIMA 모델 저장소
        self.models = {}
        self.scalers = {}
        
        # 알림 중복 방지용 캐시
        self.alert_cache = {}
        
    def load_config(self, config_path):
        """설정 파일 로드"""
        default_config = {
            'notification': {
                'email': {
                    'enabled': True,
                    'smtp_server': 'smtp.gmail.com',
                    'smtp_port': 587,
                    'sender_email': 'your_email@gmail.com',
                    'sender_password': 'your_app_password',
                    'recipient_email': 'recipient@gmail.com'
                },
                'discord': {
                    'enabled': False,
                    'webhook_url': 'YOUR_DISCORD_WEBHOOK_URL'
                },
                'slack': {
                    'enabled': False,
                    'webhook_url': 'YOUR_SLACK_WEBHOOK_URL'
                }
            },
            'alerts': {
                'price_change_threshold': 0.03,
                'volume_spike_threshold': 1.5,
                'prediction_accuracy_threshold': 0.8,
                'support_resistance_breach': True,
                'prediction_deviation_threshold': 0.05,
                'duplicate_alert_cooldown': 3600  # 1시간
            },
            'model': {
                'retrain_frequency': 7,  # 일주일마다 재훈련
                'prediction_confidence_threshold': 0.7,
                'data_period': '2y'
            }
        }
        
        if Path(config_path).exists():
            with open(config_path, 'r', encoding='utf-8') as f:
                user_config = json.load(f)
                # 기본 설정에 사용자 설정 덮어쓰기
                self.deep_update(default_config, user_config)
        
        return default_config
    
    def deep_update(self, base_dict, update_dict):
        """딕셔너리 깊은 업데이트"""
        for key, value in update_dict.items():
            if isinstance(value, dict):
                base_dict[key] = self.deep_update(base_dict.get(key, {}), value)
            else:
                base_dict[key] = value
        return base_dict
    
    def setup_logging(self):
        """로깅 설정"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('stock_alert.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def setup_database(self):
        """SQLite 데이터베이스 설정"""
        self.conn = sqlite3.connect('stock_alerts.db')
        self.conn.execute('''
            CREATE TABLE IF NOT EXISTS alerts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                alert_type TEXT NOT NULL,
                message TEXT NOT NULL,
                priority TEXT NOT NULL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        self.conn.execute('''
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                prediction_value REAL NOT NULL,
                confidence_lower REAL NOT NULL,
                confidence_upper REAL NOT NULL,
                actual_value REAL,
                prediction_date DATETIME DEFAULT CURRENT_TIMESTAMP,
                target_date DATETIME NOT NULL
            )
        ''')
        
        self.conn.commit()
    
    def get_stock_data(self, symbol, period="2y"):
        """주식 데이터 가져오기 (개선된 버전)"""
        try:
            # 한국 주식 데이터 우선 시도
            end_date = datetime.now()
            if period == '1y':
                start_date = end_date - timedelta(days=365)
            elif period == '2y':
                start_date = end_date - timedelta(days=730)
            elif period == '5y':
                start_date = end_date - timedelta(days=1825)
            else:
                start_date = end_date - timedelta(days=365)
            
            # FinanceDataReader 사용
            data = fdr.DataReader(symbol, start_date, end_date)
            
            # 주봉 데이터로 변환
            if len(data) > 500:  # 데이터가 많으면 주봉으로
                data = data.resample('W').agg({
                    'Open': 'first',
                    'High': 'max',
                    'Low': 'min',
                    'Close': 'last',
                    'Volume': 'sum'
                }).dropna()
            
            self.logger.info(f"✅ 주식 데이터 수집 완료 ({symbol}): {len(data)} 데이터 포인트")
            return data
            
        except Exception as e:
            self.logger.error(f"❌ 주식 데이터 수집 실패 ({symbol}): {e}")
            
            # 대체 방법: yfinance
            try:
                ticker = f"{symbol}.KS"
                data = yf.download(ticker, period=period, interval='1wk')
                self.logger.info(f"✅ 대체 데이터 수집 완료 ({symbol}): {len(data)} 데이터 포인트")
                return data
            except Exception as e2:
                self.logger.error(f"❌ 대체 데이터 수집도 실패 ({symbol}): {e2}")
                return None
    
    def calculate_enhanced_indicators(self, data):
        """향상된 기술적 지표 계산"""
        df = data.copy()
        
        # 기본 이동평균
        for window in [5, 20, 60]:
            df[f'MA_{window}'] = df['Close'].rolling(window=window).mean()
        
        # 볼린저 밴드
        df['BB_middle'] = df['Close'].rolling(window=20).mean()
        bb_std = df['Close'].rolling(window=20).std()
        df['BB_upper'] = df['BB_middle'] + (bb_std * 2)
        df['BB_lower'] = df['BB_middle'] - (bb_std * 2)
        df['BB_position'] = (df['Close'] - df['BB_lower']) / (df['BB_upper'] - df['BB_lower'])
        
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
        df['MACD_Histogram'] = df['MACD'] - df['MACD_Signal']
        
        # 거래량 지표
        df['Volume_MA'] = df['Volume'].rolling(window=20).mean()
        df['Volume_Ratio'] = df['Volume'] / df['Volume_MA']
        
        # 변동성 지표
        df['Price_Change'] = df['Close'].pct_change()
        df['Volatility'] = df['Price_Change'].rolling(window=20).std()
        df['ATR'] = self.calculate_atr(df)
        
        # 지지/저항선
        df['Support'] = df['Low'].rolling(window=20).min()
        df['Resistance'] = df['High'].rolling(window=20).max()
        
        return df
    
    def calculate_atr(self, df, period=14):
        """Average True Range 계산"""
        high_low = df['High'] - df['Low']
        high_close = np.abs(df['High'] - df['Close'].shift())
        low_close = np.abs(df['Low'] - df['Close'].shift())
        
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = ranges.max(axis=1)
        
        return true_range.rolling(period).mean()
    
    def build_arima_model(self, symbol):
        """ARIMA 모델 구축 (Phase 2 통합)"""
        try:
            # 데이터 가져오기
            data = self.get_stock_data(symbol, self.config['model']['data_period'])
            if data is None or len(data) < 50:
                self.logger.error(f"❌ 데이터 부족 ({symbol}): {len(data) if data is not None else 0} 포인트")
                return None
            
            # 기술적 지표 계산
            data = self.calculate_enhanced_indicators(data)
            
            # 재무 지표 시뮬레이션 (실제 환경에서는 실제 데이터 사용)
            financial_data = self.simulate_financial_data(data)
            
            # 데이터 결합
            combined_data = pd.concat([data, financial_data], axis=1)
            combined_data = combined_data.fillna(method='ffill').fillna(method='bfill')
            
            # 특성 준비
            target = combined_data['Close'].dropna()
            
            feature_columns = [
                'Volume_Ratio', 'RSI', 'MACD', 'Volatility', 'ATR',
                'BB_position', 'PER', 'ROE', 'EPS'
            ]
            
            features = combined_data[feature_columns].dropna()
            
            # 데이터 정규화
            scaler = StandardScaler()
            features_scaled = pd.DataFrame(
                scaler.fit_transform(features),
                columns=features.columns,
                index=features.index
            )
            
            # 공통 인덱스 맞추기
            common_index = target.index.intersection(features_scaled.index)
            target = target.loc[common_index]
            features_scaled = features_scaled.loc[common_index]
            
            # 정상성 확인
            if not self.check_stationarity(target):
                target = target.diff().dropna()
                features_scaled = features_scaled.iloc[1:]  # 차분으로 인한 길이 맞춤
            
            # 훈련/테스트 분할
            train_size = int(len(target) * 0.8)
            train_target = target.iloc[:train_size]
            train_features = features_scaled.iloc[:train_size]
            
            # 최적 차수 찾기
            best_order = self.find_optimal_order(train_target, train_features)
            
            # 모델 훈련
            model = ARIMA(train_target, order=best_order, exog=train_features)
            fitted_model = model.fit()
            
            # 모델 저장
            self.models[symbol] = fitted_model
            self.scalers[symbol] = scaler
            
            self.logger.info(f"✅ ARIMA 모델 구축 완료 ({symbol}): {best_order}")
            return fitted_model
            
        except Exception as e:
            self.logger.error(f"❌ ARIMA 모델 구축 실패 ({symbol}): {e}")
            return None
    
    def simulate_financial_data(self, stock_data):
        """재무 지표 시뮬레이션"""
        dates = stock_data.index
        np.random.seed(42)
        
        # 삼성전자 기반 시뮬레이션
        financial_data = pd.DataFrame({
            'PER': np.random.normal(15, 3, len(dates)),
            'PBR': np.random.normal(1.2, 0.3, len(dates)),
            'EPS': np.random.normal(5000, 1000, len(dates)),
            'ROE': np.random.normal(12, 3, len(dates)),
            'Market_Cap': np.random.normal(400, 50, len(dates))
        }, index=dates)
        
        return financial_data
    
    def check_stationarity(self, series):
        """정상성 검사"""
        try:
            result = adfuller(series.dropna())
            return result[1] <= 0.05
        except:
            return False
    
    def find_optimal_order(self, target, features, max_order=3):
        """최적 ARIMA 차수 찾기"""
        best_aic = float('inf')
        best_order = (1, 0, 1)
        
        for p in range(max_order):
            for d in range(2):
                for q in range(max_order):
                    try:
                        model = ARIMA(target, order=(p, d, q), exog=features)
                        fitted = model.fit()
                        if fitted.aic < best_aic:
                            best_aic = fitted.aic
                            best_order = (p, d, q)
                    except:
                        continue
        
        return best_order
    
    def get_prediction(self, symbol, steps=1):
        """예측 수행"""
        try:
            if symbol not in self.models:
                self.logger.warning(f"⚠️ 모델 없음 ({symbol}), 새로 구축")
                self.build_arima_model(symbol)
            
            model = self.models[symbol]
            scaler = self.scalers[symbol]
            
            # 최신 데이터로 특성 준비
            latest_data = self.get_stock_data(symbol, period="1mo")
            if latest_data is None:
                return None
            
            latest_data = self.calculate_enhanced_indicators(latest_data)
            financial_data = self.simulate_financial_data(latest_data)
            combined_data = pd.concat([latest_data, financial_data], axis=1)
            
            feature_columns = [
                'Volume_Ratio', 'RSI', 'MACD', 'Volatility', 'ATR',
                'BB_position', 'PER', 'ROE', 'EPS'
            ]
            
            features = combined_data[feature_columns].iloc[-steps:]
            features_scaled = scaler.transform(features)
            
            # 예측 수행
            forecast = model.forecast(steps=steps, exog=features_scaled)
            confidence_int = model.get_forecast(steps=steps, exog=features_scaled).conf_int()
            
            result = {
                'prediction': forecast.iloc[-1] if steps == 1 else forecast,
                'confidence_lower': confidence_int.iloc[-1, 0] if steps == 1 else confidence_int.iloc[:, 0],
                'confidence_upper': confidence_int.iloc[-1, 1] if steps == 1 else confidence_int.iloc[:, 1],
                'current_price': latest_data['Close'].iloc[-1],
                'timestamp': datetime.now()
            }
            
            # 예측 결과 저장
            self.save_prediction(symbol, result)
            
            return result
            
        except Exception as e:
            self.logger.error(f"❌ 예측 실패 ({symbol}): {e}")
            return None
    
    def save_prediction(self, symbol, prediction_result):
        """예측 결과 데이터베이스 저장"""
        try:
            self.conn.execute('''
                INSERT INTO predictions (symbol, prediction_value, confidence_lower, confidence_upper, target_date)
                VALUES (?, ?, ?, ?, ?)
            ''', (
                symbol,
                float(prediction_result['prediction']),
                float(prediction_result['confidence_lower']),
                float(prediction_result['confidence_upper']),
                (datetime.now() + timedelta(days=1)).strftime('%Y-%m-%d')
            ))
            self.conn.commit()
        except Exception as e:
            self.logger.error(f"❌ 예측 결과 저장 실패: {e}")
    
    def check_enhanced_alert_conditions(self, symbol):
        """향상된 알림 조건 확인"""
        try:
            # 최신 데이터 가져오기
            data = self.get_stock_data(symbol, period="1mo")
            if data is None or len(data) < 20:
                return []
            
            data = self.calculate_enhanced_indicators(data)
            
            # 예측 결과 가져오기
            prediction_result = self.get_prediction(symbol)
            
            alerts = []
            current_price = data['Close'].iloc[-1]
            prev_price = data['Close'].iloc[-2]
            
            # 1. 급격한 가격 변동
            price_change = abs(current_price - prev_price) / prev_price
            if price_change >= self.config['alerts']['price_change_threshold']:
                direction = "급등" if current_price > prev_price else "급락"
                alerts.append({
                    'type': 'price_change',
                    'message': f"{symbol} {direction} {price_change:.2%} (현재: {current_price:,.0f}원)",
                    'priority': 'high' if price_change > 0.05 else 'medium',
                    'data': {
                        'current_price': current_price,
                        'change_rate': price_change,
                        'direction': direction
                    }
                })
            
            # 2. 거래량 급증
            volume_ratio = data['Volume_Ratio'].iloc[-1]
            if volume_ratio >= self.config['alerts']['volume_spike_threshold']:
                alerts.append({
                    'type': 'volume_spike',
                    'message': f"{symbol} 거래량 급증 {volume_ratio:.1f}배 (현재가: {current_price:,.0f}원)",
                    'priority': 'high',
                    'data': {
                        'volume_ratio': volume_ratio,
                        'current_price': current_price
                    }
                })
            
            # 3. 기술적 지표 알림
            rsi = data['RSI'].iloc[-1]
            if rsi > 70:
                alerts.append({
                    'type': 'rsi_overbought',
                    'message': f"{symbol} RSI 과매수 구간 (RSI: {rsi:.1f})",
                    'priority': 'medium',
                    'data': {'rsi': rsi}
                })
            elif rsi < 30:
                alerts.append({
                    'type': 'rsi_oversold',
                    'message': f"{symbol} RSI 과매도 구간 (RSI: {rsi:.1f})",
                    'priority': 'medium',
                    'data': {'rsi': rsi}
                })
            
            # 4. 볼린저 밴드 돌파
            bb_position = data['BB_position'].iloc[-1]
            if bb_position > 1:
                alerts.append({
                    'type': 'bb_breakout_upper',
                    'message': f"{symbol} 볼린저 밴드 상단 돌파 (현재: {current_price:,.0f}원)",
                    'priority': 'high',
                    'data': {'bb_position': bb_position}
                })
            elif bb_position < 0:
                alerts.append({
                    'type': 'bb_breakout_lower',
                    'message': f"{symbol} 볼린저 밴드 하단 이탈 (현재: {current_price:,.0f}원)",
                    'priority': 'high',
                    'data': {'bb_position': bb_position}
                })
            
            # 5. 예측 대비 실제 가격 괴리
            if prediction_result:
                prediction_deviation = abs(current_price - prediction_result['prediction']) / current_price
                if prediction_deviation >= self.config['alerts']['prediction_deviation_threshold']:
                    alerts.append({
                        'type': 'prediction_deviation',
                        'message': f"{symbol} AI 예측 대비 {prediction_deviation:.2%} 괴리 (예측: {prediction_result['prediction']:,.0f}원, 실제: {current_price:,.0f}원)",
                        'priority': 'medium',
                        'data': {
                            'prediction': prediction_result['prediction'],
                            'actual': current_price,
                            'deviation': prediction_deviation
                        }
                    })
            
            # 6. 지지/저항선 돌파
            support = data['Support'].iloc[-1]
            resistance = data['Resistance'].iloc[-1]
            
            if current_price > resistance:
                alerts.append({
                    'type': 'resistance_break',
                    'message': f"{symbol} 저항선 돌파 (저항: {resistance:,.0f}원 → 현재: {current_price:,.0f}원)",
                    'priority': 'high',
                    'data': {'resistance': resistance, 'current_price': current_price}
                })
            elif current_price < support:
                alerts.append({
                    'type': 'support_break',
                    'message': f"{symbol} 지지선 이탈 (지지: {support:,.0f}원 → 현재: {current_price:,.0f}원)",
                    'priority': 'high',
                    'data': {'support': support, 'current_price': current_price}
                })
            
            # 중복 알림 필터링
            filtered_alerts = self.filter_duplicate_alerts(symbol, alerts)
            
            return filtered_alerts
            
        except Exception as e:
            self.logger.error(f"❌ 알림 조건 확인 실패 ({symbol}): {e}")
            return []
    
    def filter_duplicate_alerts(self, symbol, alerts):
        """중복 알림 필터링"""
        filtered = []
        current_time = datetime.now()
        cooldown = self.config['alerts']['duplicate_alert_cooldown']
        
        for alert in alerts:
            cache_key = f"{symbol}_{alert['type']}"
            
            if cache_key in self.alert_cache:
                last_alert_time = self.alert_cache[cache_key]
                if (current_time - last_alert_time).seconds < cooldown:
                    continue  # 쿨다운 기간 내의 중복 알림 제외
            
            self.alert_cache[cache_key] = current_time
            filtered.append(alert)
        
        return filtered
    
    def send_enhanced_notification(self, symbol, alerts):
        """향상된 알림 발송"""
        if not alerts:
            return
        
        try:
            # 데이터베이스에 알림 기록
            for alert in alerts:
                self.conn.execute('''
                    INSERT INTO alerts (symbol, alert_type, message, priority)
                    VALUES (?, ?, ?, ?)
                ''', (symbol, alert['type'], alert['message'], alert['priority']))
            self.conn.commit()
            
            # 각 채널로 알림 발송
            if self.config['notification']['email']['enabled']:
                self.send_email_notification(symbol, alerts)
            
            if self.config['notification']['discord']['enabled']:
                self.send_discord_notification(symbol, alerts)
            
            if self.config['notification']['slack']['enabled']:
                self.send_slack_notification(symbol, alerts)
            
            self.logger.info(f"📨 알림 발송 완료 ({symbol}): {len(alerts)}개 알림")
            
        except Exception as e:
            self.logger.error(f"❌ 알림 발송 실패 ({symbol}): {e}")
    
    def send_email_notification(self, symbol, alerts):
        """이메일 알림 발송 (개선된 버전)"""
        try:
            config = self.config['notification']['email']
            
            msg = MIMEMultipart()
            msg['From'] = config['sender_email']
            msg['To'] = config['recipient_email']
            msg['Subject'] = f"🔔 [{symbol}] 주식 알림 {len(alerts)}건"
            
            # HTML 형식 이메일 본문
            html_body = f"""
            <html>
            <body>
                <h2>📊 {symbol} 주식 알림</h2>
                <p><strong>알림 시간:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                <hr>
            """
            
            for alert in alerts:
                priority_color = "#FF0000" if alert['priority'] == 'high' else "#FFA500"
                priority_icon = "🔴" if alert['priority'] == 'high' else "🟡"
                
                html_body += f"""
                <div style="margin: 10px 0; padding: 10px; border-left: 4px solid {priority_color};">
                    <p><strong>{priority_icon} {alert['message']}</strong></p>
                </div>
                """
            
            html_body += """
                <hr>
                <p><small>이 알림은 AI 기반 주식 분석 시스템에서 자동 생성되었습니다.</small></p>
            </body>
            </html>
            """
            
            msg.attach(MIMEText(html_body, 'html'))
            
            server = smtplib.SMTP(config['smtp_server'], config['smtp_port'])
            server.starttls()
            server.login(config['sender_email'], config['sender_password'])
            server.send_message(msg)
            server.quit()
            
            self.logger.info(f"📧 이메일 알림 발송 완료 ({symbol})")