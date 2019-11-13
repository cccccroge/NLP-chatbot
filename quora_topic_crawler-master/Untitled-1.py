import pickle
infile = open("Typical Days_pickle",'rb')
new_dict = pickle.load(infile)
infile.close()
print(new_dict)