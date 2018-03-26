import pickle as pk

mylist = pk.load(open('Error_file/perfect.p', 'rb'))
for k in range(mylist.shape[0]):
    for i in range(mylist.shape[1]):
        for j in range(mylist.shape[2]):
            if i-1 == j:
                mylist[k][i][j] = 0.002
            elif i == j:
                mylist[k][i][j] = 0.998
            elif i+1 == j:
                mylist[k][i][j] = 1.000
            else:
                mylist[k][i][j] = 0
print(mylist[9])
pk.dump(mylist, open('Error_file/996.p', 'wb'))
