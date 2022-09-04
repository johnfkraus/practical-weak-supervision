import pickle


def unpickling_data(filename):
    file = open(filename,'rb')
    new_data = pickle.load(file)
    file.close()
    return new_data


filename = 'PersonalInfo'

filename2 = 'numbers'

print(unpickling_data(filename2))