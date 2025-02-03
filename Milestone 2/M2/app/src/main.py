from db import save_to_db, save_lookup
import os
from cleaning import clean
from cleaning import save
from cleaning import load
import pandas as pd
from run_producer import start_producer, stop_container
from consumer import run_consumer
import time


if __name__ == '__main__':
  data_dir='data/'
  df=load(data_dir)

  cleaned_csv_path = data_dir + 'fintech_data_Met_P1_52_23665_clean.csv'
  lookup_csv_path = data_dir + 'lookup_table_Met_P1_52_23665.csv'
  if os.path.exists(cleaned_csv_path) and os.path.exists(lookup_csv_path):
      cleaned_data = pd.read_csv(cleaned_csv_path)
      lookup_table = pd.read_csv(lookup_csv_path)
  else:
      df = load(data_dir)
      cleaned_data ,lookup_table= clean(df)
      save(data_dir, cleaned_data)
  save_to_db(cleaned_data)
  save_lookup(lookup_table)

  time.sleep(1)
  id=start_producer("52_23665","fintech",)
  run_consumer()
  stop_container(id)
  print('> Done!')
