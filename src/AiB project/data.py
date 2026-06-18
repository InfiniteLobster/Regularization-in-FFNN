#--------------------Libraries--------------------#
#
#--------------------Functions--------------------#
#Basic data loading handler.
def load_data():
    FOLDS = ['c000', 'c001', 'c002', 'c003', 'c004', 'f000', 'f001', 'f002', 'f003', 'f004']
    alleles = os.listdir("dataset")
    allele_folds = np.concatenate(
        list(map(lambda x: np.loadtxt(f"dataset/{x}/{FOLDS[0]}", dtype=str), alleles))
    )
    return allele_folds


#Basic train/test split handler.
def train_test_split(data, percentage):
    split_train = int(data.shape[0] * (1 - percentage))
    split_test  = int(data.shape[0] * percentage)
    return data[:split_train], data[-split_test:]

dataset = load_data()
train, test = train_test_split(dataset, percentage=0.2)
print(train.shape, test.shape)
