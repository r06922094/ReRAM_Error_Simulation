import sys
import pickle as pk
import random
if len(sys.argv) < 2:
    print('Usage: python3 generate_table.py [error_list.pkl]')
    exit()

error_list = pk.load(open(sys.argv[1], 'rb'))

err = error_list
m = []

for u in range(10):
    m.append([])
    for i in range(11):
        m[u].append([0]*int(err[u][i][0]*100.))
        for j in range(1, 11):
            m[u][i] += [j]*int((err[u][i][j]-err[u][i][j-1])*100.)
        m[u][i] += [i] * (100-len(m[u][i]))
        random.shuffle(m[u][i])
print(m[0])

pk.dump(m, open('table.p', 'wb'))


