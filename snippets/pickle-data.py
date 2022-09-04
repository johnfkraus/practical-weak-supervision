import pickle

def pickle_data(data, filename):
    # data = {
    #             'name': 'Prashant',
    #             'profession': 'Software Engineer',
    #             'country': 'India'
    #     }

    outfile = open(filename, 'wb')
    pickle.dump(data,outfile)
    outfile.close()


data = {
    'name': 'Prashant',
    'profession': 'Software Engineer',
    'country': 'India'
}

filename = 'numbers'

data2 = [1,2,3,4]

pickle_data(data2, filename)

