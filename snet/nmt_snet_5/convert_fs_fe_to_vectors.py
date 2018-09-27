f = ["feature_s","feature_e"]
dtype = ["_dev","_train","_test"]
suffix = [".source"]

for i in f:
	for j in dtype:
		file = i + j + suffix
		with open(file) as ff, open(file+"_new","w") as fn:
			line = ff.readline()
			while line:
