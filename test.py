import sys


args = sys.argv[:]
print(type(args))
print(args)
bands = args[3:]
print(type(bands))
bands = list(map(int,bands))
bands = list(map(minus,bands))
print(bands)

def minus(item):
	return item-1