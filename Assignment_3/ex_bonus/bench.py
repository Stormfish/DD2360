import subprocess

#for particles in range(100000,1100000,100000):
	#print("particles:"+str(particles)+"\n")
	#for blocksize in (2**p for p in range(5,10)):
sizes = [64, 128, 256, 512, 1024, 2048, 4096] 
for s in sizes:
	args = ['./bonus','-s'+str(s)]
	subprocess.call(args) 		
