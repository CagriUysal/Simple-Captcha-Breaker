with open('/home/yasin/test.txt') as f:
	numbers = [[int(j) for j in list(i)] for i in f.read().splitlines()]
digits_dir = 'digits/'

for cap_i in range(1, len(numbers)+1):
	for j in range(6):
		digit_name = digits_dir + 'digit_' + str(cap_i+1000) + '_' + str(j) + '.jpeg'
		target_dir = str(numbers[cap_i-1][j])
		print 'mv %s %s' % (digit_name, target_dir)
