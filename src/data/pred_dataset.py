import os
import numpy as np
import pandas as pd
import os
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
import wfdb
import ast
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold

from src.data.timefeatures import time_features
import warnings

warnings.filterwarnings('ignore')


class Dataset_ETT_hour(Dataset):
    def __init__(self, root_path, split='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, timeenc=0, freq='h',
                 use_time_features=False
                 ):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert split in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[split]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        self.use_time_features = use_time_features

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))
        print("df_raw columns:", df_raw.columns)

        border1s = [0, 12 * 30 * 24 - self.seq_len, 12 * 30 * 24 + 4 * 30 * 24 - self.seq_len]
        border2s = [12 * 30 * 24, 12 * 30 * 24 + 4 * 30 * 24, 12 * 30 * 24 + 8 * 30 * 24]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            data_stamp = df_stamp.drop(['date'], axis=1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        if self.use_time_features: return _torch(seq_x, seq_y, seq_x_mark, seq_y_mark)
        else: return _torch(seq_x, seq_y)

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_ETT_minute(Dataset):
    def __init__(self, root_path, split='train', size=None,
                 features='S', data_path='ETTm1.csv',
                 target='OT', scale=True, timeenc=0, freq='t',
                 use_time_features=False
                 ):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert split in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[split]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        self.use_time_features = use_time_features

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))
        print("df_raw columns:", df_raw.columns)

        border1s = [0, 12 * 30 * 24 * 4 - self.seq_len, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4 - self.seq_len]
        border2s = [12 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 8 * 30 * 24 * 4]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            df_stamp['minute'] = df_stamp.date.apply(lambda row: row.minute, 1)
            df_stamp['minute'] = df_stamp.minute.map(lambda x: x // 15)
            data_stamp = df_stamp.drop(['date'], axis=1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        if self.use_time_features: return _torch(seq_x, seq_y, seq_x_mark, seq_y_mark)
        else: return _torch(seq_x, seq_y)

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)

class Dataset_PTB_XL(Dataset):
    def __init__(self, root_path, split='train', size=None,
                 data_path='ptbxl_database_cleaned.csv', target='scp_codes',
                 train_split=0.7, test_split=0.2, scale=True):
        print(">>> [PTB_XL] __init__ started")
        assert split in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[split]

        self.target = target
        self.train_split = train_split
        self.test_split = test_split
        self.scale = scale

        self.root_path = root_path
        self.data_path = data_path
        self.min_val = None
        self.max_val = None

        print(">>> [PTB_XL] Reading CSV...")
        df_all = pd.read_csv(os.path.join(self.root_path, self.data_path))
        print(">>> [PTB_XL] CSV loaded. Shape:", df_all.shape)

        # 1. Find all unique codes
        print(">>> [PTB_XL] Finding all unique codes...")
        all_codes = set()
        for codestr in df_all['scp_codes']:
            d = ast.literal_eval(codestr)
            all_codes.update(d.keys())
        all_codes = sorted(list(all_codes))  # consistent order
        print(">>> [PTB_XL] Found", len(all_codes), "unique codes.")

        # 2. Count support for each code
        print(">>> [PTB_XL] Counting support for each code...")
        code_counts = {code: 0 for code in all_codes}
        for codestr in df_all['scp_codes']:
            d = ast.literal_eval(codestr)
            for code in d.keys():
                code_counts[code] += 1
        print(">>> [PTB_XL] Support counted.")

        min_support = 5  # or your chosen threshold
        rare_codes = [code for code, count in code_counts.items() if count <= min_support]
        common_codes = [code for code in all_codes if code not in rare_codes]
        print(">>> [PTB_XL] Rare codes being removed:", rare_codes)
        print(">>> [PTB_XL] Number of common codes:", len(common_codes))
        self.all_codes = common_codes  # Only keep common codes

        # 3. Build label matrix using only common codes
        print(">>> [PTB_XL] Building label matrix...")
        label_matrix = []
        for codestr in df_all['scp_codes']:
            d = ast.literal_eval(codestr)
            label_matrix.append([d.get(code, 0.0) for code in self.all_codes])
        label_matrix = np.array(label_matrix)
        print(">>> [PTB_XL] Label matrix shape:", label_matrix.shape)

        # 4. Stratified split
        print(">>> [PTB_XL] Performing stratified split...")
        mskf = MultilabelStratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        train_idx, val_idx = next(mskf.split(np.zeros(len(label_matrix)), label_matrix))
        print(">>> [PTB_XL] Stratified split done. Train size:", len(train_idx), "Val size:", len(val_idx))

        if self.set_type == 0:  # train
            self.data = df_all.iloc[train_idx].reset_index(drop=True)
            print(">>> [PTB_XL] Assigned train data. Shape:", self.data.shape)
        elif self.set_type == 1:  # val
            self.data = df_all.iloc[val_idx].reset_index(drop=True)
            print(">>> [PTB_XL] Assigned val data. Shape:", self.data.shape)
        else:  # test
            self.data = df_all.iloc[val_idx].reset_index(drop=True)  # or your test logic
            print(">>> [PTB_XL] Assigned test data. Shape:", self.data.shape)

        print(">>> [PTB_XL] __init__ complete")

    def __read_data__(self):
        df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path))
        print("df_raw columns:", df_raw.columns)
        # Select only the id and target columns
        df_raw = df_raw[['ecg_id', 'filename_hr', self.target]]

        # Drop duplicate columns by name (keep the first occurrence)
        df_raw = df_raw.loc[:, ~df_raw.columns.duplicated()]

        # Now continue as before
        num_train = int(len(df_raw) * self.train_split)
        num_test = int(len(df_raw) * self.test_split)
        num_vali = len(df_raw) - num_train - num_test
        border1s = [0, num_train, num_train + num_vali]
        border2s = [num_train, num_train + num_vali, len(df_raw)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]
        self.data = df_raw.iloc[border1:border2].reset_index(drop=True)

        # No need for further duplicate checks!
        self.data = self.data[self.data['filename_hr'].astype(str).str.startswith('records500/')].reset_index(drop=True)
        self.data = self.data[self.data['filename_hr'] != 'filename_hr'].reset_index(drop=True)
        print("First 5 filename_hr values:", self.data['filename_hr'].head())

        # Scale the time-series data if scaling is enabled
        if self.scale:
            all_time_series = []
            for filename_hr in self.data['filename_hr']:
                file_path = os.path.join(self.root_path, filename_hr)
                signals, _ = wfdb.rdsamp(file_path)
                all_time_series.append(signals)
            if self.set_type == 0:
                numeric_data = np.vstack(all_time_series)
                self.min_val = numeric_data.min(axis=0)
                self.max_val = numeric_data.max(axis=0)

    def __getitem__(self, index):
        ecg_id = self.data.iloc[index]['ecg_id']
        filename_hr = self.data.iloc[index]['filename_hr']
        codestr = self.data.iloc[index][self.target]

        # Load the time-series data from the file
        file_path = os.path.join(self.root_path, filename_hr)
        signals, _ = wfdb.rdsamp(file_path)

        # Scale the time-series data if scaling is enabled
        if self.scale and self.min_val is not None and self.max_val is not None:
            signals = (signals - self.min_val) / (self.max_val - self.min_val)
        signals = np.array(signals)

        # --- NEW: Parse scp_codes into a multi-label vector ---
        code_dict = ast.literal_eval(codestr)

        target_vec = np.array([code_dict.get(code, 0.0) for code in self.all_codes], dtype=np.float32)

        return torch.tensor(signals, dtype=torch.float32), torch.tensor(target_vec, dtype=torch.float32)

    def __len__(self):
        return len(self.data)

    def inverse_transform(self, scaled_data):
        if not self.scale or self.min_val is None or self.max_val is None:
            raise ValueError("Scaling is not enabled or min/max values are not set.")
        return scaled_data * (self.max_val - self.min_val) + self.min_val

class Dataset_Custom(Dataset):
    def __init__(self, root_path, split='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, timeenc=0, freq='h',
                 time_col_name='date', use_time_features=False, 
                 train_split=0.7, test_split=0.2
                 ):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        
        # init
        assert split in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[split]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        self.time_col_name = time_col_name
        self.use_time_features = use_time_features

        # train test ratio
        self.train_split, self.test_split = train_split, test_split

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))
        print("df_raw columns:", df_raw.columns)
        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.select_dtypes(include=[np.number]).columns
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]
        '''
        df_raw.columns: [time_col_name, ...(other features), target feature]
        '''
        cols = list(df_raw.columns)
        #cols.remove(self.target) if self.target
        #cols.remove(self.time_col_name)
        #df_raw = df_raw[[self.time_col_name] + cols + [self.target]]
        
        num_train = int(len(df_raw) * self.train_split)
        num_test = int(len(df_raw) * self.test_split)
        num_vali = len(df_raw) - num_train - num_test
        border1s = [0, num_train - self.seq_len, len(df_raw) - num_test - self.seq_len]
        border2s = [num_train, num_train + num_vali, len(df_raw)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[[self.time_col_name]][border1:border2]
        df_stamp[self.time_col_name] = pd.to_datetime(df_stamp[self.time_col_name])
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp[self.time_col_name].apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp[self.time_col_name].apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp[self.time_col_name].apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp[self.time_col_name].apply(lambda row: row.hour, 1)
            data_stamp = df_stamp.drop([self.time_col_name], axis=1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp[self.time_col_name].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        if self.use_time_features: return _torch(seq_x, seq_y, seq_x_mark, seq_y_mark)
        else: return _torch(seq_x, seq_y)

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)
    

class Dataset_Pred(Dataset):
    def __init__(self, root_path, split='pred', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, inverse=False, timeenc=0, freq='15min', cols=None):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert split in ['pred']

        self.features = features
        self.target = target
        self.scale = scale
        self.inverse = inverse
        self.timeenc = timeenc
        self.freq = freq
        self.cols = cols
        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))
        print("df_raw columns:", df_raw.columns)
        '''
        df_raw.columns: ['date', ...(other features), target feature]
        '''
        if self.cols:
            cols = self.cols.copy()
            cols.remove(self.target)
        else:
            cols = list(df_raw.columns)
            cols.remove(self.target)
            cols.remove('date')
        df_raw = df_raw[['date'] + cols + [self.target]]
        border1 = len(df_raw) - self.seq_len
        border2 = len(df_raw)

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            self.scaler.fit(df_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        tmp_stamp = df_raw[['date']][border1:border2]
        tmp_stamp['date'] = pd.to_datetime(tmp_stamp.date)
        pred_dates = pd.date_range(tmp_stamp.date.values[-1], periods=self.pred_len + 1, freq=self.freq)

        df_stamp = pd.DataFrame(columns=['date'])
        df_stamp.date = list(tmp_stamp.date.values) + list(pred_dates[1:])
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            df_stamp['minute'] = df_stamp.date.apply(lambda row: row.minute, 1)
            df_stamp['minute'] = df_stamp.minute.map(lambda x: x // 15)
            data_stamp = df_stamp.drop(['date'], axis=1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        if self.inverse:
            self.data_y = df_data.values[border1:border2]
        else:
            self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        if self.inverse:
            seq_y = self.data_x[r_begin:r_begin + self.label_len]
        else:
            seq_y = self.data_y[r_begin:r_begin + self.label_len]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


def _torch(*dfs):
    return tuple(torch.from_numpy(x).float() for x in dfs)
