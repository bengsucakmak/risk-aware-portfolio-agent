import pandas as pd
import numpy as np
import yfinance as yf
from finrl.meta.preprocessor.preprocessors import FeatureEngineer
import config

class DataProcessor:
    def __init__(self):
        self.start_date = config.TRAIN_START_DATE
        self.end_date = config.TEST_END_DATE
        self.ticker_list = config.TICKER_LIST
        self.indicators = config.INDICATORS

    def download_data(self):
        """
        FinRL'in kendi indiricisi yerine daha sağlam bir manuel indirme fonksiyonu.
        Hatalı hisseleri atlar ve formatı garantiye alır.
        """
        data_df = pd.DataFrame()
        
        for tic in self.ticker_list:
            try:
                # Veriyi indir
                temp_df = yf.download(tic, start=self.start_date, end=self.end_date, progress=False)
                
                # Eğer veri boşsa atla
                if temp_df.empty:
                    print(f"UYARI: {tic} için veri bulunamadı, atlanıyor.")
                    continue

                # Veri formatını düzenle
                temp_df = temp_df.reset_index()
                
                # Sütun isimlerini düzelt (YFinance bazen MultiIndex döndürür)
                try:
                    # Sütun isimlerini string'e çevir
                    temp_df.columns = temp_df.columns.map(''.join) if isinstance(temp_df.columns, pd.MultiIndex) else temp_df.columns
                    # Ticker adını sütunlardan temizle
                    temp_df.columns = [col.replace(tic, "") if tic in col else col for col in temp_df.columns]
                except:
                    pass

                # FinRL formatına uygun hale getirme
                df_formatted = pd.DataFrame()
                df_formatted['date'] = temp_df.iloc[:, 0] # Tarih
                
                # Sütun isimlerini küçük harfe çevir
                temp_df.columns = [str(c).lower() for c in temp_df.columns]
                
                df_formatted['open'] = temp_df['open']
                df_formatted['high'] = temp_df['high']
                df_formatted['low'] = temp_df['low']
                
                if 'adj close' in temp_df.columns:
                     df_formatted['close'] = temp_df['adj close']
                else:
                     df_formatted['close'] = temp_df['close']
                     
                df_formatted['volume'] = temp_df['volume']
                df_formatted['tic'] = tic
                
                data_df = pd.concat([data_df, df_formatted], axis=0)
                print(f"{tic} başarıyla indirildi.")
                
            except Exception as e:
                print(f"HATA: {tic} indirilirken sorun oluştu: {e}")

        data_df['date'] = data_df['date'].astype(str)
        data_df = data_df.sort_values(['date', 'tic']).reset_index(drop=True)
        data_df['day'] = pd.to_datetime(data_df['date']).dt.day_name().apply(lambda x: str(x)[:3].upper())
        
        return data_df

    def add_indicators(self, df):
        print("Teknik indikatörler hesaplanıyor...")
        fe = FeatureEngineer(
            use_technical_indicator=True,
            tech_indicator_list=self.indicators,
            use_vix=False,        # <--- BURAYI FALSE YAPTIK (HATAYI ÇÖZEN KISIM)
            use_turbulence=True,  # Türbülans endeksi kalabilir, o indirme yapmaz.
            user_defined_feature=False
        )
        df_processed = fe.preprocess_data(df)
        return df_processed

    def clean_data(self, df):
        df = df.copy()
        df = df.fillna(0)
        df = df.replace(np.inf, 0)
        df = df.sort_values(['date', 'tic']).reset_index(drop=True)
        return df

    def run(self):
        print("--- Veri İndirme Başlıyor ---")
        raw_df = self.download_data()
        
        print(f"\nToplam İndirilen Veri Satırı: {len(raw_df)}")
        
        if len(raw_df) == 0:
            raise ValueError("Hiçbir veri indirilemedi!")

        print("--- Özellik Mühendisliği (Feature Engineering) ---")
        processed_df = self.add_indicators(raw_df)
        
        print("--- Veri Temizleme ---")
        final_df = self.clean_data(processed_df)
        
        final_df.to_csv(config.DATA_SAVE_PATH, index=False)
        print(f"Veri başarıyla kaydedildi: {config.DATA_SAVE_PATH}")
        print("Sütunlar:", final_df.columns.tolist())
        return final_df