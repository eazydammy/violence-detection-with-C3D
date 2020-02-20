import os

paths = []
with open('listOfFiles.txt', 'r') as file:
    for line in file:
        path = str(line)[:-1]
        dirListing = os.listdir(path)
        paths.append([path, len(dirListing)])

with open('train_list.txt', 'w') as file:
    for item in paths:
        path = item[0]
        if path[54] == 'n':
            label = '0'
        else:
            label = '1'
        length = item[1]
        n = length // 8
        for i in range(n-1):
            file.write(path+' '+str(8*i+1)+' '+label+'\n')