
import numpy as np
import random
import matplotlib.pyplot as plt
import os
from mpi4py import MPI
from wapylmp import MyLammps


from parameters import (
  data_file_mol_flat,data_file_slab_flat,ff_filepath_morse,
  template_output_dir,test_dir,at,at_mol,bt,bt_mol,agt,agt_mol,dt,it,
  DeltaEnergy,Sol_RandSeed,Sol_Temperature,Sol_DampTemp,Liq_Temperature,Liq_DampTemp,NumSteps_EQ1,
  bxl,bxh,byl,byh,bzl,bzh
  )


	# ----------------------PSO parameters setting ---------------------------------

itr_pso=0
Num=0
D0=np.array([20.0]) #ester only (values based on O_amide-Fe)
alpha=np.array([2.0])
r0=np.array([1.8])



class PSO():

	def __init__(self, pN, dim, max_iter):

		self.w = 0.9
		self.c1 = 0.5
		self.c2 = 0.4
		self.r1 = random.random()
		self.r2 = random.random()
		self.pN = pN								# particle number
		self.dim = dim								# search dimension
		self.max_iter = max_iter  
		self.X = np.zeros((self.pN, self.dim))		# all particle Coordinate 
		self.V = np.zeros((self.pN, self.dim))
		self.pbest = np.zeros((self.pN, self.dim))  #  the best Coordinate of each particle
		self.gbest = np.zeros((1, self.dim)) 		# the best Coordinate of all particle(global)
		self.p_fit = np.zeros(self.pN) 				# the best fitness of each particle
		self.fit = 1e10 							# the best fitness of all particle (global)
		#the refer values and its dimension
		self.goal_flat = np.array([1376.492108,366.9530513,41.65784122,-56.23269992,-73.72874113,-65.89332883,-52.78580205,-40.59595499,-30.74108055])
		self.e_ad_best_flat=np.zeros(9)
		#weight setting 
		self.w_max=0.9
		self.w_min=0.1
		self.u=np.array([0,1,1,1,2,1,1,1,1])
		self.g_best_array =[]
		self.gb_Num=np.zeros((1,1))
		self.g_b_Num_array=[]
		self.check_array=[]
		random.seed(42)

	# ---------------------Target function-----------------------------

	def function(self,E_ad_np_flat):
		a = np.sum((self.u)*((E_ad_np_flat/self.goal_flat-1)**2))   
		return a

	# ---------------------Initialization group----------------------------------
	def init_Population(self):
		for i in range(self.pN):
			for j in range(self.dim):
				if (j<1):
					self.X[i][j] = random.uniform(5.0, 25.0)
					self.V[i][j] = random.uniform(0, 100.0)
				elif(j<2):
					self.X[i][j] = random.uniform(1.5, 3.0)
					self.V[i][j] = random.uniform(0, 100.0)
				else :
					self.X[i][j] = random.uniform(1.5, 3.0)
					self.V[i][j] = random.uniform(0, 100.0)
			
			#run first time energy#
			D0[0]=self.X[i][0]
			alpha[0]=self.X[i][1]
			r0[0]=self.X[i][2]
			
			#flat			
			runlammps(itr_pso,i,D0,alpha,r0,200)
			for k in np.array([-2,-1.75,-1.5,-1.25,-1.0,-0.75,-0.5,-0.25,0]):
				runlammps(itr_pso,i,D0,alpha,r0,k)
			E_ad_np_flat=calculate_E_ad(itr_pso,i)
			
			self.pbest[i] = self.X[i]
			tmp = self.function(E_ad_np_flat)
			self.p_fit[i] = tmp
			if tmp < self.fit:
				self.fit = tmp
				self.gbest = self.X[i]
				self.e_ad_best_flat = E_ad_np_flat
				self.check_array.append(D0[0])
				self.check_array.append(alpha[0])
				self.check_array.append(r0[0])
				self.check_array.append(8888) #no means
				with open('log_pso_0.txt', 'a') as file:
					file.write('Initialisation adsorption energies:' + '\n' + str(self.e_ad_best_flat) + '\n' + 'Morse ' + str(self.gbest) + '\n'+ '\n')
				

	# ----------------------Update coordinates----------------------------------

	def iterator(self):

		fitness = []		

		for t in range(self.max_iter):
			self.w=self.w_max-(self.w_max-self.w_min)*(t+1)/self.max_iter

			for i in range(self.pN):
				
				D0[0]=self.X[i][0]
				alpha[0]=self.X[i][1]
				r0[0]=self.X[i][2]
				
				#flat			
				runlammps(t+1,i,D0,alpha,r0,200)
				for k in np.array([-2,-1.75,-1.5,-1.25,-1.0,-0.75,-0.5,-0.25,0]):
					runlammps(t+1,i,D0,alpha,r0,k)
				E_ad_np_flat=calculate_E_ad(t+1,i)

				temp = self.function(E_ad_np_flat)
				if temp < self.p_fit[i]:  # update the best fitness of each particle
					self.p_fit[i] = temp
					self.pbest[i] = self.X[i]
					if self.p_fit[i] < self.fit:  # update the best fitness of all particle(global)
						self.gbest = self.X[i]
						self.fit = self.p_fit[i]
						self.e_ad_best_flat = E_ad_np_flat
						self.gb_Num=i
						self.check_array.append(D0[0])
						self.check_array.append(alpha[0])
						self.check_array.append(r0[0])
						self.check_array.append(8888)		
						with open('log_pso_0.txt', 'a') as file:
							file.write('Best global fitness= ' + str(self.fit) + '\n')
							file.write('Best global fitness at particle index  ' + str(self.gb_Num) + '\n')
							file.write("Best global position (Morse)= " + str(self.gbest) + '\n')
							file.write("Adsorption energies= " + str(self.e_ad_best_flat) + '\n')
							file.write("%i -th iteration weight %s\n" % (t, self.w) + '\n')	
						fitness.append(self.fit)
						self.g_best_array.append(self.gbest)
						self.g_b_Num_array.append(self.gb_Num)

			for i in range(self.pN):
				self.V[i] = self.w * self.V[i] + self.c1 * self.r1 * (self.pbest[i] - self.X[i]) + self.c2 * self.r2 * (self.gbest - self.X[i])
				self.X[i] = self.X[i] + self.V[i]

				#reset the parameter values
				self.X[self.X<0.02]=random.uniform(0.02, 5)
				#self.X[self.X>20]=random.uniform(0.01, 20)
				p=np.random.random()
				if p >0.7:
					self.X[i][0]=random.uniform(5.0, 25.0)
					#self.X[i][1]=random.uniform(0.01, 100.0)
					self.X[i][1]=random.uniform(1.5, 3.0)
					#self.X[i][3]=random.uniform(0.01, 10.0)
					self.X[i][2]=random.uniform(1.5, 3.0)
					#self.X[i][5]=random.uniform(0.01, 2.0)

			
			
			# with open('log_pso_0.txt', 'a') as file:
			# 	file.write('Best global fitness= ' + str(self.fit) + '\n')
			# 	file.write('Best global fitness at particle index  ' + str(self.gb_Num) + '\n')
			# 	file.write("Best global position (Morse)= " + str(self.gbest) + '\n')
			# 	file.write("Adsorption energies= " + str(self.e_ad_best_flat) + '\n')
			# 	file.write("%i -th iteration weight %s\n" % (t, self.w) + '\n')
			
			# print(self.X[0], end=" ")
			# print(self.fit)  
			# print("%i time iterator self.w\n"%(t))
			# print(self.w)

		# with open('log_pso_0.txt', 'a') as file:
		# 	file.write("Final global position (Morse)= " + str(self.gbest) + '\n')
		# 	file.write("Final adsorption energies= " + str(self.e_ad_best_flat) + '\n')

		with open('log_pso_0.txt', 'a') as file:
			file.write("Final global position (Morse)= " + str(self.g_best_array) + '\n')

		# print("final time iterator gbest:\n")
		# print(self.gbest)		
		# print("%i time self.e_ad_best\n"%(t))
		# print(self.e_ad_best_flat)
		
		
		plt.plot(range(len(fitness)), fitness)
		plt.xlabel("Iterations")
		plt.ylabel("Global Best Fitness Value")
		plt.savefig('fitness_plot_0.png')
	
		return fitness




# ----------------------LAMMPS related-----------------------
def runlammps(itr_pso,Num,D0,alpha,r0,sz):
	output_dir = template_output_dir.format(itr_pso)
	if MPI.COMM_WORLD.rank == 0:
		if not os.path.isdir(output_dir):
			os.makedirs(output_dir)

	L = MyLammps(name="trans")
	L.processors("*","*",1)
	L.units("real")
	L.atom_style("full")
	L.dimension(3)
	L.boundary("p","p","p")
	L.region("mybox","block",bxl,bxh,byl,byh,bzl,bzh)
	L.create_box(at,"mybox","bond/types",bt,"angle/types",agt,"dihedral/types",dt,"improper/types",it,"extra/bond/per/atom",1000,"extra/angle/per/atom",1000,"extra/dihedral/per/atom",1000,"extra/improper/per/atom",1000,"extra/special/per/atom",1000 )
	
	#=== Forcefield Files ===#
	L.pair_style("hybrid/overlay morse",10.0,"lj/class2/coul/long",10.5)
	L.bond_style("class2")
	L.angle_style("class2")
	L.dihedral_style("class2")
	L.improper_style("class2")

	with open(ff_filepath_morse, "r") as f:
		for line in f.read().split("\n"):
			L.command(line)
	L.log(os.path.join(output_dir, "log%s_%s"%(itr_pso,Num)),"append")				 
	L.pair_modify("pair lj/class2/coul/long shift yes")
	L.pair_modify("pair morse shift yes")
	L.pair_coeff(1,13,"morse",D0[0],alpha[0],r0[0])                                     #O_ester-Fe
	L.pair_coeff(7,13,"morse",19.293705410180948,1.8284233432420596,2.005117341768946)  #O_radical-Fe D0[0],alpha[0],r0[0] 
		
	#=== Data Files ===#
	
	L.read_data(data_file_slab_flat,"add append offset",at_mol,bt_mol,agt_mol,0,0)
	L.read_data(data_file_mol_flat,"add append shift",0,0,sz)
	
	#=== Simulation Settings ===#
	L.special_bonds("lj/coul",0.0,0.0,1.0)
	L.kspace_style("pppm",DeltaEnergy)
	# L.kspace_modify("slab", 3.0)

	#=== Setting Solid ===#
	L.group(" lb type", 9    )    # Lower Base
	L.group(" lt type", 10 ,11)     # Lower Thermostat
	L.group(" li type", 12 ,13)    # Lower Interface
	L.neigh_modify(" exclude group lb lb")
	L.neigh_modify(" exclude group lb lt")
	L.neigh_modify(" exclude group lb li")
	L.neigh_modify(" exclude group lt lt")
	L.neigh_modify(" exclude group lt li")
	L.neigh_modify(" exclude group li li")

	#=== Monitor ===#	
	
	L.thermo_style("custom pe ebond eangle edihed eimp evdwl etail ecoul elong epair")
	L.run(NumSteps_EQ1)
	L.write_data("data.estertempo_check_%s"%(sz))


def calculate_E_ad(itr_pso,Num):
	rf = open("./lmp_results/itr_%s/log%s_%s"%(itr_pso,itr_pso,Num),"r")
	all = rf.readlines()
	allstr = "".join(all).split()
	a=[i for i,val in enumerate(allstr) if val=='PotEng']
	eng=[]
	E_ad=[]
	for i in a:
		eng.append(allstr[i+10])
	for i in eng[1:]:
		E_ad.append(float(i)-float(eng[0]))
	E_ad_np=np.array(E_ad)
	return E_ad_np


if __name__ == "__main__":

	my_pso = PSO(pN=100, dim=3, max_iter=25)
	my_pso.init_Population()
	fitness = my_pso.iterator()
	print("final fitness\n")
	print(fitness)
	print("final time iterator gbest:\n")
	print(my_pso.gbest)
	
	path_pso = os.path.join(test_dir, "restart_flat_Morse_test_set_{}.txt".format(0))
	with open(path_pso, "w") as f:
		f.write("# test\n")
		f.write("final time iterator gbest:\n")
		f.write(str(my_pso.gbest))
		f.write("\nfinal\n")
		f.write(str(fitness))
		f.write("\nfinal time self.e_ad_best_flat\n")
		f.write(str(my_pso.e_ad_best_flat))
		f.write("\nfinal time self.g_best_array\n")
		f.write(str(my_pso.g_best_array))
		f.write("\nfinal time self.g_b_Num_array\n")
		f.write(str(my_pso.g_b_Num_array))
		f.write("\nfinal time self.check_array\n")
		f.write(str(my_pso.check_array))


