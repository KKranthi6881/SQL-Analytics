# File: airflow/includes/sql/transform/green_taxi_transform.py

import pandas as pd
import sqlite3
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

class GreenTaxiTransformer:
    def __init__(self, source_db: str, target_db: str):
        self.source_db = source_db
        self.target_db = target_db

    def _convert_datetime_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convert datetime columns with robust error handling"""
        if df.empty:
            return df
            
        datetime_columns = ['pickup_datetime', 'dropoff_datetime']
        for col in datetime_columns:
            if col in df.columns:
                try:
                    # First attempt: Try direct datetime conversion
                    df[col] = pd.to_datetime(df[col], errors='coerce')
                    
                    # If we have any null values, try decoding bytes
                    if df[col].isna().any():
                        temp_series = df[col].copy()
                        byte_mask = temp_series.apply(lambda x: isinstance(x, bytes))
                        
                        if byte_mask.any():
                            # Try different encodings for bytes
                            encodings = ['utf-8', 'latin1', 'cp1252']
                            for encoding in encodings:
                                try:
                                    temp_series[byte_mask] = temp_series[byte_mask].apply(
                                        lambda x: x.decode(encoding) if isinstance(x, bytes) else x
                                    )
                                except Exception as e:
                                    continue
                            
                            # Convert decoded strings to datetime
                            df[col] = pd.to_datetime(temp_series, errors='coerce')
                    
                    logger.info(f"Successfully converted {col} to datetime")
                    logger.info(f"Sample values from {col}: {df[col].head()}")
                    logger.info(f"Null count in {col}: {df[col].isna().sum()}")
                    
                except Exception as e:
                    logger.error(f"Error converting {col} to datetime: {str(e)}")
                    raise
                
        return df

    def transform_and_load(self):
        """Transform green taxi data and load into fact table"""
        query = """
        SELECT 
            vendor_id,
            pickup_datetime,
            dropoff_datetime,
            passenger_count,
            trip_distance,
            pickup_locationid,
            dropoff_locationid,
            rate_code_id,
            payment_type,
            fare_amount,
            extra,
            mta_tax,
            tip_amount,
            tolls_amount,
            total_amount,
            ehail_fee,
            trip_type,
            improvement_surcharge,
            congestion_surcharge
        FROM staging_green_taxi
        """
        
        try:
            with sqlite3.connect(self.source_db) as source_conn:
                source_conn.text_factory = str
                df = pd.read_sql(query, source_conn)
                
            if df.empty:
                logger.warning("No green taxi data found")
                return
                
            logger.info(f"Initial data shape: {df.shape}")
            logger.info(f"Columns and their types: {df.dtypes}")
            
            # Convert datetime columns
            df = self._convert_datetime_columns(df)
            
            # Get time_id values for pickup and dropoff
            # Note: In a real implementation, you would look these up from dim_time
            # For now, we'll use NULL as we'll need to coordinate with dimension tables
            df['pickup_time_id'] = None
            df['dropoff_time_id'] = None
            
            # Calculate trip duration
            duration_mask = df['dropoff_datetime'].notna() & df['pickup_datetime'].notna()
            df['trip_duration_minutes'] = 0
            df.loc[duration_mask, 'trip_duration_minutes'] = (
                (df.loc[duration_mask, 'dropoff_datetime'] - 
                 df.loc[duration_mask, 'pickup_datetime'])
                .dt.total_seconds() / 60
            )
            
            # Calculate speed
            df['avg_speed_mph'] = 0.0
            valid_duration = (
                (df['trip_duration_minutes'] > 0) & 
                df['trip_distance'].notna() & 
                (df['trip_distance'] > 0)
            )
            df.loc[valid_duration, 'avg_speed_mph'] = (
                df.loc[valid_duration, 'trip_distance'] / 
                (df.loc[valid_duration, 'trip_duration_minutes'] / 60)
            )
            
            # Handle extra charges
            df['extra_charges'] = (
                df['extra'].fillna(0) + 
                df['mta_tax'].fillna(0) + 
                df['tolls_amount'].fillna(0) + 
                df['improvement_surcharge'].fillna(0) + 
                df['congestion_surcharge'].fillna(0)
            )
            
            # Add service type
            df['service_type'] = 'green'
            
            # Select and rename columns to match fact_trips schema
            fact_trips_columns = {
                'vendor_id': 'vendor_id',
                'pickup_time_id': 'pickup_time_id',
                'dropoff_time_id': 'dropoff_time_id',
                'pickup_locationid': 'pickup_location_id',
                'dropoff_locationid': 'dropoff_location_id',
                'passenger_count': 'passenger_count',
                'trip_distance': 'trip_distance',
                'rate_code_id': 'rate_code_id',
                'payment_type': 'payment_type_id',
                'fare_amount': 'fare_amount',
                'extra_charges': 'extra_charges',
                'tip_amount': 'tip_amount',
                'total_amount': 'total_amount',
                'trip_duration_minutes': 'trip_duration_minutes',
                'avg_speed_mph': 'avg_speed_mph',
                'service_type': 'service_type'
            }
            
            # Create final DataFrame with correct columns
            df_final = pd.DataFrame()
            for source_col, target_col in fact_trips_columns.items():
                df_final[target_col] = df[source_col]
            
            # Clean up data
            df_final = df_final.replace([float('inf'), -float('inf')], 0)
            df_final = df_final.fillna(0)
            
            logger.info(f"Final data shape: {df_final.shape}")
            logger.info(f"Final columns: {df_final.columns.tolist()}")
            
            # Load into fact_trips
            with sqlite3.connect(self.target_db) as target_conn:
                df_final.to_sql('fact_trips', target_conn, if_exists='append', index=False)
                
            logger.info(f"Successfully processed {len(df_final)} green taxi records")
            
        except Exception as e:
            logger.error(f"Error processing green taxi data: {str(e)}")
            raise

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    source_db = "/opt/airflow/database/sqlite/nyc_taxi.db"
    target_db = "/opt/airflow/database/sqlite/nyc_datawarehouse.db"
    transformer = GreenTaxiTransformer(source_db, target_db)
    transformer.transform_and_load()
    # adding few comments
