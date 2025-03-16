
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import datetime as datetime
from collections import Counter
import time
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split

db_green = pd.read_parquet("green_taxi_data.parquet")
db_yellow = pd.read_parquet("yellow_taxi_data.parquet")

db_green.head()
print(db_green)