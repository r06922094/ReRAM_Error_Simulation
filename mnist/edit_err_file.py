import pickle as pk

mylist = pk.load(open('old/Error_file/perfect.p', 'rb'))
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
mylist[0][1][0] = 1.0
mylist[0][1][1] = 0
print(mylist[0])
pk.dump(mylist, open('perfect.p', 'wb'))
