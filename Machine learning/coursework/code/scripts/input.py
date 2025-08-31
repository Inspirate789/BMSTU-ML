import numpy as np
from kalman import filter
import os.path

def read_rssi_data(dist):
    filenames = [f'new/{dist}m_3.txt', f'new/{dist}m_2.txt', f'new/{dist}m.txt']
    for filename in filenames:
        if os.path.exists(filename):
            data = [int(line.strip()) for line in open(filename, 'r')]
            input_len = len(data)
            data = [val for val in data if val != 127]
            lost_count = input_len-len(data)
            loss = round(100*lost_count/len(data), 2)
            print(f'\tdist {dist}m\t{filename}\t{len(data)} entries + {lost_count} invalid ({loss}%)')
            return np.array(data), lost_count
    print(filenames)

def read_input():
    dist_range = np.arange(0.1, 5+0.001, 0.1)
    rssi = []
    rssi_filtered = []
    lost_count_total = 0

    rssi_min, rssi_max = None, None
    print('input:')

    for dist in dist_range:
        data, lost_count = read_rssi_data(round(dist, 1))
        lost_count_total += lost_count
        filtered_data = filter(data)

        rssi.append(data)
        rssi_filtered.append(filtered_data)
        
        if not rssi_min or min(data) < rssi_min:
            rssi_min = min(data)
        if not rssi_max or max(data) > rssi_max:
            rssi_max = max(data)
    
    
    total_length = sum([len(data) for data in rssi])
    loss = round(100*lost_count_total/total_length, 2)
    print(f'total: {total_length} entries + {lost_count_total} invalid ({loss}%)')

    rssi_min, rssi_max = rssi_min - 10, rssi_max + 10
    rssi_range = np.arange(rssi_min, rssi_max + 1)

    filtered_length = sum([len(data) for data in rssi_filtered])
    rssi_filtered = [[val for val in row if rssi_min <= val <= rssi_max] for row in rssi_filtered]
    new_filtered_length = sum([len(data) for data in rssi_filtered])
    loss = round(100*(filtered_length-new_filtered_length)/filtered_length, 2)
    print(f'after filtration: {new_filtered_length} entries + {filtered_length-new_filtered_length} invalid ({loss}%)')

    return rssi, rssi_filtered, dist_range, rssi_range
