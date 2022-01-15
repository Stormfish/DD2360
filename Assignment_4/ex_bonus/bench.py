import subprocess

#for particles in range(100000,1100000,100000):
	#print("particles:"+str(particles)+"\n")
	#for blocksize in (2**p for p in range(5,10)):
#particles=10000000
#						   1000000
for particles in range(1,10000001,1000000):
	#print(particles)
	block = 256
	args = ['./ex_bonus','-s'+str(particles), '-b'+str(block)]
	subprocess.call(args) 