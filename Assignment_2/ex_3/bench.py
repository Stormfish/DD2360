import subprocess

#for particles in range(100000,1100000,100000):
	#print("particles:"+str(particles)+"\n")
	#for blocksize in (2**p for p in range(5,10)):
particles=10000000
blocksize=32
	#print("blocksize:"+str(blocksize)+"\n")
args = ['./ex_3','-s'+str(particles), '-b'+str(blocksize)]
subprocess.call(args) 