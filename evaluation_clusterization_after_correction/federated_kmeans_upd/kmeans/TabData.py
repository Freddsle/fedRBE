import numpy as np
import pandas as pd
import copy

class TabData:
    def __init__(self, data, columns, rows, scaled):
        self.data = data
        self.columns = columns
        self.rows = rows
        self.scaled = scaled
        self.col_count = data.shape[1]

    @classmethod
    def from_file(cls, filename, header=0, index=0, sep ='\t', keep_original=True, federated_dimension='columns') -> 'TabData':
        """

        :param filename:
        :param header:
        :param index:
        :param sep:
        :return:
        """

        data = pd.read_csv(filepath_or_buffer=filename, header=header, sep=sep, index_col=index)
        if data.shape[1] == 0:
            print('Suspiciously few columns ... using sniffer.')
            data = pd.read_csv(filepath_or_buffer=filename, header=header, sep=None, index_col=index, engine='python')

        sample_ids = np.array(data.index)
        variable_names = np.array(data.columns.values)
        if keep_original:
            data = data.values
            scaled = copy.deepcopy(data)
        else:
            scaled = data.values
            data = None

        #make sure the federated dimension are the columns.
        if federated_dimension == 'rows':
            data = data.T
            scaled = scaled.T
            return cls(data, sample_ids, variable_names, scaled)
        else:
            return cls(data, variable_names, sample_ids, scaled)
