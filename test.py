import timeit

t1 = timeit.default_timer()

for i in range(10):
    for j in range(10):
        for k in range(10):
            print(i,j,k)

t2 = timeit.default_timer()

print(t2-t1)