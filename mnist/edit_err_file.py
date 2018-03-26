import pickle as pk

mylist = pk.load(open('perfect.p', 'rb'))
for k in range(mylist.shape[0]):
    for i in range(mylist.shape[1]):
        for j in range(mylist.shape[2]):
            if i-1 == j:
                mylist[k][i][j] = 0
            elif i == j:
                mylist[k][i][j] = 1.0
            elif i+1 == j:
                mylist[k][i][j] = 0
            else:
                mylist[k][i][j] = 0
#print(mylist[3])
pk.dump(mylist, open('perfect.p', 'wb'))
