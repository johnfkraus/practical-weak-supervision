import pickle

carlist = ['Toyota','Honda','BMW','Audi', 'Suzuki']
print(carlist)

# with open ("carlist.pkl","wb") as carpickle:
#    pickle.dump(list,carpickle)

# unpickling the list
with open("carlist.pkl","rb") as carpickle:
   mycar=pickle.load(carpickle)

print(mycar)

