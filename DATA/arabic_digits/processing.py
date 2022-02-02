import numpy as np

def load_from_txt(mode):
    path =  mode+'_Arabic_Digit.txt'
    with open(path) as f:
        contents = f.read()

    data = contents.split("\n            \n")
    data_list = []
    for seq_string in data:
        seq = np.fromstring(seq_string, sep='\n')
        seq = seq.reshape(-1, 13)
        data_list.append(seq)
    return data_list


def pad_data(data,mode):
    if mode == 'Train':
        label = np.arange(10).reshape(10, 1).repeat(660, 1).reshape(-1)
    else:
        label = np.arange(10).reshape(10, 1).repeat(220, 1).reshape(-1)
    seq_list = []
    del_list = []
    for i in range(len(data)):
        if (data[i].shape[0] >= 20) * (data[i].shape[0] <= 50):
            seq_list.append(data[i])
        else:
            del_list.append(i)
    label = np.delete(label, del_list, 0)

    seq_array = []
    for seq in seq_list:
        seq_pad = np.zeros((50, 13))
        seq_pad[:seq.shape[0]] = seq
        seq_array.append(seq_pad)
    seq_array = np.array(seq_array)
    np.save(mode.lower() + '_array.npy', seq_array)
    np.save(mode.lower() + '_label.npy', label)

if __name__ == "__main__":
    for mode in ['Test','Train']:
        data = load_from_txt(mode)
        pad_data(data,mode)