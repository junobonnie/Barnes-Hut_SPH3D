# -*- coding: utf-8 -*-
"""
Created on Wed Jul 26 19:14:08 2023

@author: replica
"""

from vectortools3D import *
from atom3D import *
import sys

L = 20 # smoothing length
K = 50*0.02*L**6 # isentropic exponent
#alpha = 0.5
rho_0 = 0.0
mu = 0.0002*L**6

#================

limit_number = 15000
sys.setrecursionlimit(limit_number)

softening_length = 10 #5.5
theta_ = 0.5

class Render(Render):
    def __init__(self, screen, width, height, depth, angle=(0,0,0), omega = (0,0,0), focus_factor = 1):
        pg.init()
        self.screen = screen
        self.width = width
        self.height = height
        self.depth = depth
        self.focus_factor = focus_factor
        self.render_vector = Vector(0, height, 0)
        self.render_metric = Tensor(1, 0, 0, 
                                    0, -1, 0,
                                    0, 0, 1)
        self.origin_vector = Vector(width/2, height/2, depth/2)
        self.angle = angle
        self.omega = omega
        self.t = 0

    def atom(self, atom):
        self.circle(atom.pos, atom.element.radius, atom.color)
        
class Atom(Atom):
    def __init__(self, element, pos, vel = Vector(0, 0, 0), rho = 0, p = 0, color = (0,0,0)):
        self.element = element
        self.pos = pos
        self.vel = vel
        self.rho = rho
        self.p = p   
        self.color = color
        
    def kinetic_energy(self):
        return (1/2)*self.element.mass*self.vel.dot(self.vel)
        
    def potential_energy(self, other):
        r = self.pos - other.pos
        if not self == other:# and r.dot(r) > self.element.radius+other.element.radius:
            return -self.element.mass*other.element.mass/m.sqrt(r.dot(r)+softening_length**2)
        else:
            return 0
        
    def fusion(self, other_Atom):
        new_Atom = None
        if not self == other_Atom:
            d = self.pos - other_Atom.pos
            if (d.dot(d) < (self.element.radius + other_Atom.element.radius)**2):
                new_element = Element(name = 'New atom', mass = self.element.mass + other_Atom.element.mass, 
                                      radius = m.sqrt(self.element.radius**2 + other_Atom.element.radius**2),
                                      color = self.element.color + other_Atom.element.color)
                new_Atom = Atom(element = new_element, 
                                pos = (self.element.mass*self.pos + other_Atom.element.mass*other_Atom.pos)/(self.element.mass + other_Atom.element.mass),
                                vel = (self.element.mass*self.vel + other_Atom.element.mass*other_Atom.vel)/(self.element.mass + other_Atom.element.mass))
        return new_Atom
    
class World(World):
    def __init__(self, t, atoms, G, gravity = Vector(0, 0, 0)):
        self.t = t
        self.atoms = atoms
        self.G = G
        self.gravity = gravity

class Simulator(Simulator):
    def __init__(self, dt, world, render, grid_size = 100):
        self.dt = dt
        self.world = world
        self.render = render
        self.count_screen = 0
        self.count_snapshot = 0
        self.grid_size = grid_size
        self.grid = None
        
    def set_density(self):
        for atom in self.world.atoms:
            result = 0
            atoms = self.get_near_atoms(atom)
            for other in atoms:
                r = atom.pos - other.pos
                if not atom == other and r.dot(r) < L**2:# and r.dot(r) > self.element.radius+other.element.radius:
                    result += (15*atom.element.mass/(3.14*L**6))*((L - abs(r))**3) #-other.element.mass/r.dot(r)*(r/abs(r))
            atom.rho = result
            #print('============')
            #print(atom.rho)
    
    def set_pressure(self):
        for atom in self.world.atoms:
            atom.p = max(K*(atom.rho-rho_0),0)
            #print(atom.rho)
    
    def pressure_acc(self, atom): 
        result = Vector(0, 0, 0) 
        atoms = self.get_near_atoms(atom)
        for other in atoms:
            r = atom.pos - other.pos
            if not atom == other and r.dot(r) < L**2: # and r.dot(r) > atom.element.radius+other.element.radius:
                #print(other.rho)
                result += (45/(3.14*L**6))*(r/(abs(r)+0.001))*((atom.p + other.p)/(2*other.rho+0.002))*((L-abs(r))**2) #-other.element.mass/r.dot(r)*(r/abs(r))
                #print(result)
        return result
    #Pᵢ = (− (45 M) / (π L⁶)) ∑ⱼ − (xⱼ − xᵢ) / (dᵢⱼ) (pⱼ + pᵢ) / (2 ρⱼ) (L − dᵢⱼ)² 
    
    def viscosity_acc(self, atom):
        result = Vector(0, 0, 0)
        atoms = self.get_near_atoms(atom)
        for other in atoms:
            r = atom.pos - other.pos
            if not atom == other and r.dot(r) < L**2:# and r.dot(r) > self.element.radius+other.element.radius:
                result += (45*mu/(3.14*L**6))*((other.vel - atom.vel)/(other.rho+0.001))*(L-abs(r)) #-other.element.mass/r.dot(r)*(r/abs(r))
                #result += (15*mu/(2*3.14*L**3))*((other.vel - atom.vel)/(other.rho+0.01))*((-1.5*(r.dot(r)/L**3)+(2*abs(r)/(L**2))+(L/(2*r.dot(r))))) #-other.element.mass/r.dot(r)*(r/abs(r))
        return result
    #Vᵢ = (45 μ M) / (π L⁶) ∑ⱼ (uⱼ − uᵢ) / (ρⱼ) (L − dᵢⱼ)
        
   # Function to build the Barnes-Hut tree
    def build_tree(self, atoms, x, y, z, region_size):
        if len(atoms) == 0:
            return None
    
        if len(atoms) == 1:
            # draw cube
            draw_cube = False
            #draw_cube = True
            if draw_cube:
                self.render.cube(Vector(x, y, z), region_size, region_size, region_size, (255,0,0))
            return atoms[0]
    
        # Calculate center of mass and total mass for the combined atoms
        total_mass = 0
        center = Vector(0, 0, 0)
        for atom in atoms:
            total_mass += atom.element.mass
            center += atom.element.mass*atom.pos
        center /= total_mass
    
        # Divide the atoms into quadrants
        cube_atoms = [[[[], []], [[], []]], [[[], []], [[], []]]]
        for atom in atoms:
            x_index = 0 if atom.pos.x < x+region_size/2 else 1
            y_index = 0 if atom.pos.y < y+region_size/2 else 1
            z_index = 0 if atom.pos.z < z+region_size/2 else 1
            cube_atoms[x_index][y_index][z_index].append(atom)
    
        # Recursively build the tree
        cube_ = [[[], []], [[], []]]
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    cube_[i][j].append(self.build_tree(cube_atoms[i][j][k], x+i*region_size/2, y+j*region_size/2, z+k*region_size/2, region_size/2))
        return [cube_, center, total_mass, region_size]
 
    def calculate_force(self, atom1, atom2):
        r = atom1.pos - atom2.pos
        f = -self.world.G*atom2.element.mass*r/((r.dot(r)+softening_length**2)**(3/2))
        return f
  
    def calculate_net_force(self, atom, tree):
        f = Vector(0, 0, 0)
        if tree is None:
            return f

        if isinstance(tree, Atom):
            if tree != atom:
                force = self.calculate_force(atom, tree)
                f += force
        else:
            d = abs(tree[1] - atom.pos)
            if  tree[3] < theta_*d:
                force = self.calculate_force(atom, Atom(pos = tree[1], element=Element('tree', tree[2], 1, 'red')))
                f += force
            else:
                for i in range(2):
                    for j in range(2):
                        for k in range(2):
                            f += self.calculate_net_force(atom, tree[0][i][j][k])
        return f
    
    def get_region_size(self):
        region_size = 0
        for atom in self.world.atoms:
            for p in atom.pos.list():
                region_size = max(region_size, 2*abs(p))
        return region_size
                
    def main(self):
        region_size = self.get_region_size()
        tree = self.build_tree(self.world.atoms, -region_size/2, -region_size/2, -region_size/2, region_size)
        x_ = []
        v_ = []
        self.make_grid()
        self.set_density()
        self.set_pressure()
        for atom in self.world.atoms:
            force = self.calculate_net_force(atom, tree)
            new_v = atom.vel + self.pressure_acc(atom)*self.dt + self.viscosity_acc(atom)*self.dt + force*self.dt + self.world.gravity*self.dt
            v_.append(new_v)
            x_.append(atom.pos + new_v*self.dt)
        
        count = 0
        for atom in self.world.atoms:
            if atom.element.name == 'Helium':
                atom.color = (0,0,max(255-int(5*255*atom.rho),0))
            else:
                atom.color = (max(255-int(5*255*atom.rho),0),0,0)
            atom.pos = x_[count]
            atom.vel = v_[count]
            count = count + 1
        
def recursive_safety(atoms):
    for atom in atoms:
        for other_atom in atoms:
            if not atom == other_atom:
                d = atom.pos - other_atom.pos
                if d.dot(d) < 10:
                    atoms.remove(other_atom)
                    
if __name__ == '__main__':
    DEBUG = False
        
    width = 1000
    height = 1000
    depth = 1000

    screen = pg.display.set_mode((width, height))
    render = Render(screen, width, height, depth, angle=(1,0,0), omega = (0,0,0))
    clock = pg.time.Clock()

    black = pg.Color('black')
    white = pg.Color('white')
    red = pg.Color('red')
    green = pg.Color('green')
    blue = pg.Color('blue')

    e1 = Element(name = 'Helium', mass = 10, radius = 3, color = blue)
    e2 = Element(name = 'Uranium', mass = 10, radius = 3, color = red)

    # atom1 = Atom(e1, Vector(-200, 0), Vector(50, 0))
    # atom2 = Atom(e1, Vector(0, 0))
    # atom3 = Atom(e1, Vector(25, -10))
    # atom4 = Atom(e1, Vector(25, 10))
    # atom5 = Atom(e1, Vector(50, -20))
    # atom6 = Atom(e1, Vector(50, 0))
    # atom7 = Atom(e1, Vector(50, 20))

    walls = [] # [wall1, wall2, wall3, wall4]#, wall5]
    atoms = [] # [atom1, atom2, atom3, atom4, atom5, atom6, atom7]
    
    import random as r
    import math as m
    
    # for i in range(2000):
    #     xV = r.randrange(-400, 400, 2*e1.radius)
    #     rV = Vector(xV, r.randrange(-50, 50, 2*e1.radius))
    #     if xV < 0:
    #         atoms.append(Atom(e1, rV, Vector(0, 50)))
    #     else:
    #         atoms.append(Atom(e1, rV, -Vector(0, 50)))
    
    ################################ 
    
    for i in range(5000):
        theta = r.random()*2*m.pi
        rV = SO3_z(theta).dot(Vector(r.randrange(0, 200, 2*e1.radius) ,0 , -10*r.random())) - 0*Vector(250, 0, 0)
        atoms.append(Atom(e1, rV, abs((rV + 0*Vector(250, 0, 0))/200)*10*3*SO3_z(theta).dot(Vector(0, 20, 0)) + 0*Vector(250, 0, 0)))
    
    ################################    
    
    # for i in range(1000):
    #     theta = r.random()*2*m.pi
    #     rV = SO2(theta).dot(Vector(r.randrange(0, 50, 2*e1.radius) ,0)) - Vector(350, 20)
    #     atoms.append(Atom(e1, rV, 3*SO2(theta).dot(Vector(0, 30)) + Vector(400,0)))
    
    # for i in range(1000):
    #     theta = r.random()*2*m.pi
    #     rV = SO2(theta).dot(Vector(r.randrange(0, 50, 2*e2.radius) ,0)) + Vector(350, 20)
    #     atoms.append(Atom(e2, rV, 3*SO2(theta).dot(Vector(0, 30)) - Vector(400,0)))
        
    recursive_safety(atoms)
    
    G = 1000
    gravity = Vector(0, 0, 0)

    world = World(0, atoms, G, gravity)

    simulator = Simulator(0.01, world, render, L)
    if DEBUG:  
        t_list = []
        K_list = []
        P_list = []
        TOT_E_list = []
    
    while True:
        t = simulator.clock()
        simulator.render.update_time(t)
        simulator.draw_background(white)
        #simulator.draw_grid(100)
        simulator.main()
        # simulator.atom_wall_collision()
        # simulator.atom_atom_collision()
        # simulator.atom_atom_fusion()
        simulator.draw_atom()

        # render.text('pos = (%.2f, %.2f)'%(atoms[0].pos.x, atoms[0].pos.y) , None, 30, Vector(atoms[0].pos.x -100, atoms[0].pos.y - 30), black)
        # render.text('vel = (%.2f, %.2f)'%(atoms[0].vel.x, atoms[0].vel.y) , None, 30, Vector(atoms[0].pos.x -100, atoms[0].pos.y - 50), black)

        # render.text('pos = (%.2f, %.2f)'%(atoms[50].pos.x, atoms[50].pos.y) , None, 30, Vector(atoms[50].pos.x -100, atoms[50].pos.y - 30), blue)
        # render.text('vel = (%.2f, %.2f)'%(atoms[50].vel.x, atoms[50].vel.y) , None, 30, Vector(atoms[50].pos.x -100, atoms[50].pos.y - 50), blue)
        
        if DEBUG:   
            K = 0
            P = 0
         
            for atom in atoms:
                K = K + atom.kinetic_energy()
                for other_atom in atoms:
                    P = P + world.G*atom.potential_energy(other_atom)
            P = P/2
                
            t_list.append(t)
            K_list.append(K)
            P_list.append(P)
            TOT_E_list.append(K+P)
            
            render.text('t = %.2f'%(t) , None, 30, Vector(-480, -270), red)
            render.text('K_E = %.2f'%(K) , None, 30, Vector(-480, -300), red)
            render.text('P_E = %.2f'%(P) , None, 30, Vector(-480, -330), red)
            render.text('TOT_E = %.2f'%(K+P) , None, 30, Vector(-480, -360), red)
    
        for event in pg.event.get():
            if event.type == pg.QUIT:
                sys.exit()
        clock.tick(100)
        pg.display.update()
        
        # you need 'images/Barnes_Hut_demo_1' directory path
        #simulator.save_screen('images/Barnes_Hut_SPH_demo_1')
        
        # if t > 10:
        #     break
        
    if DEBUG:      
        import matplotlib.pyplot as plt
        
        plt.figure(figsize = (10,10))
        plt.plot(t_list, K_list, color='blue', label = 'Kinetic energy')
        plt.plot(t_list, P_list, color='orange', label = 'Potential energy')
        plt.plot(t_list, TOT_E_list, label = 'Total energy')
        plt.xlabel('time')
        plt.ylabel('energy')
        plt.axhline(sum(K_list)/len(t_list), 0, max(t_list), color='blue', linestyle='--', linewidth='1', label = 'Kinetic energy avg')
        plt.axhline(sum(P_list)/len(t_list), 0, max(t_list), color='orange', linestyle='--', linewidth='1', label = 'Potential energy avg')
        plt.legend(loc = 'best')
        plt.savefig('energy.png')
        plt.show()
        plt.close()
        
        mass_list = []
        kinetic_energy_list = []
        speed_list = []
        e1_speed_list = []
        e2_speed_list = []
        r_list = []
        e1_r_list = []
        e2_r_list = []
        
        for atom in atoms:
            mass_list.append(atom.element.mass)
            kinetic_energy_list.append(atom.kinetic_energy())
            speed_list.append(abs(atom.vel))
            r_list.append(abs(atom.pos))
            
            if atom.element.name == e1.name:
                e1_speed_list.append(abs(atom.vel))
                e1_r_list.append(abs(atom.pos))
                
            elif atom.element.name == e2.name:
                e2_speed_list.append(abs(atom.vel))
                e2_r_list.append(abs(atom.pos))
            
        plt.hist(mass_list, bins = 50)
        plt.xlabel('mass')
        plt.savefig('mass.png')
        plt.show()
        plt.close()
        
        plt.hist(kinetic_energy_list, bins = 50)
        plt.xlabel('kinetic energy')
        plt.savefig('kinetic.png')
        plt.show()
        plt.close()
        
        plt.hist(speed_list, bins = 50, label = 'Total', alpha = 0.5)
        plt.hist(e1_speed_list, bins = 50, label = 'e1', alpha = 0.5)
        plt.hist(e2_speed_list, bins = 50, label = 'e2', alpha = 0.5)
        plt.xlabel('speed')
        plt.legend(loc = 'best')
        plt.savefig('speed.png')
        plt.show()
        plt.close()
        
        plt.hist(r_list, bins = 50, label = 'Total', alpha = 0.5)
        plt.hist(e1_r_list, bins = 50, label = 'e1', alpha = 0.5)
        plt.hist(e2_r_list, bins = 50, label = 'e2', alpha = 0.5)
        plt.xlabel('distance')
        plt.legend(loc = 'best')
        plt.savefig('distance.png')
        plt.show()
        plt.close()