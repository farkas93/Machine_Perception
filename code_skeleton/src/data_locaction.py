dataconfig = dict()

# Store your possible data locations here
dataconfig['localfolder'] = 'K:/MLData/MP2020_data'
dataconfig['leonhard'] = '/cluster/project/infk/hilliges/lectures/mp20/project3'

# Set the data location for training
dataconfig['location'] = dataconfig['leonhard']

# Set datalocation for training, test and validation dataset
dataconfig['train_data'] = dataconfig['location'] + 'mp20_train.h5'
dataconfig['test_data'] = dataconfig['location'] + 'mp20_test_students.h5'
dataconfig['val_data'] = dataconfig['location'] + 'mp20_validation.h5'