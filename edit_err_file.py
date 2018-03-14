import pickle as pk

mylist = pk.load(open('Err_file.p', 'rb'))
for i in range(mylist.shape[1]):
    for j in range(mylist.shape[2]):
        if i-1 == j:
            mylist[9][i][j] = 0
        elif i == j:
            mylist[9][i][j] = 1.00
        elif i+1 == j:
            mylist[9][i][j] = 0
        else:
            mylist[9][i][j] = 0
print(mylist[9])
pk.dump(mylist, open('Err_file.p', 'wb'))
