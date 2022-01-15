import subprocess

#for particles in range(100000,1100000,100000):
	#print("particles:"+str(particles)+"\n")
	#for blocksize in (2**p for p in range(5,10)):
#particles=10000000
#						   1000000
for particles in range(0,410000000,10000000):
	#print(particles)
	args = ['./ex_3','-s'+str(particles)]
	subprocess.call(args) 