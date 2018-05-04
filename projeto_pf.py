#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Esta classe deve conter todas as suas implementações relevantes para seu filtro de partículas
"""

from pf import Particle, create_particles, draw_random_sample
import numpy as np
import inspercles # necessário para o a função nb_lidar que simula o laser
import math
from scipy.stats import norm

largura = 775 # largura do mapa
altura = 748  # altura do mapa

# Robo
robot = Particle(largura/2, altura/2, math.pi/4, 1.0)

# Nuvem de particulas
particulas = []

num_particulas = 10


# Os angulos em que o robo simulado vai ter sensores
angles = np.linspace(0.0, 2*math.pi, num=8, endpoint=False)

# Lista mais longa
movimentos_longos = [[-10, -10, 0], [-10, 10, 0], [-10,0,0], [-10, 0, 0],
              [0,0,math.pi/12.0], [0, 0, math.pi/12.0], [0, 0, math.pi/12],[0,0,-math.pi/4],
              [-5, 0, 0],[-5,0,0], [-5,0,0], [-10,0,0],[-10,0,0], [-10,0,0],[-10,0,0],[-10,0,0],[-15,0,0],
              [0,0,-math.pi/4],[0, 10, 0], [0,10,0], [0, 10, 0], [0,10,0], [0,0,math.pi/8], [0,10,0], [0,10,0],
              [0,10,0], [0,10,0], [0,10,0],[0,10,0],
              [0,0,-math.radians(90)],
              [math.cos(math.pi/3)*10, math.sin(math.pi/3),0],[math.cos(math.pi/3)*10, math.sin(math.pi/3),0],[math.cos(math.pi/3)*10, math.sin(math.pi/3),0],
              [math.cos(math.pi/3)*10, math.sin(math.pi/3),0]]

# Lista curta
movimentos_curtos = [[-10, -10, 0], [-10, 10, 0], [-10,0,0], [-10, 0, 0]]

movimentos_relativos = [[0, -math.pi/3],[10, 0],[10, 0], [10, 0], [10, 0],[15, 0],[15, 0],[15, 0],[0, -math.pi/2],[10, 0],
                       [10,0], [10, 0], [10, 0], [10, 0], [10, 0], [10, 0],
                       [10,0], [10, 0], [10, 0], [10, 0], [10, 0], [10, 0],
                       [10,0], [10, 0], [10, 0], [10, 0], [10, 0], [10, 0],
                       [0, -math.pi/2],
                       [10,0], [10, 0], [10, 0], [10, 0], [10, 0], [10, 0],
                       [10,0], [10, 0], [10, 0], [10, 0], [10, 0], [10, 0],
                       [10,0], [10, 0], [10, 0], [10, 0], [10, 0], [10, 0],
                       [10,0], [10, 0], [10, 0], [10, 0], [10, 0], [10, 0],
                       [0, -math.pi/2],
                       [10,0], [0, -math.pi/4], [10,0], [10,0], [10,0],
                       [10,0], [10, 0], [10, 0], [10, 0], [10, 0], [10, 0],
                       [10,0], [10, 0], [10, 0], [10, 0], [10, 0], [10, 0]]



movimentos = movimentos_relativos



def cria_particulas(minx=0, miny=0, maxx=largura, maxy=altura, n_particulas=num_particulas):
    """
        Cria uma lista de partículas distribuídas de forma uniforme entre minx, miny, maxx e maxy
    """

    return create_particles(robot.pose(),((maxx-minx)/2),((maxy-miny)/2),np.pi,n_particulas)

def move_particulas(particulas, movimento):
    """
        Recebe um movimento na forma [deslocamento, theta]  e o aplica a todas as partículas
        Assumindo um desvio padrão para cada um dos valores
        Esta função não precisa devolver nada, e sim alterar as partículas recebidas.

        Sugestão: aplicar move_relative(movimento) a cada partícula

        Você não precisa mover o robô. O código fornecido pelos professores fará isso

    """
        
    for particula in particulas:
        speed = [norm.rvs(movimento[0],1.5),norm.rvs(movimento[1],np.radians(1))]
        particula.move_relative(speed)

    return particulas

def leituras_laser_evidencias(robot, particulas):
    """
        Realiza leituras simuladas do laser para o robo e as particulas
        Depois incorpora a evidência calculando
        P(H|D) para todas as particulas
        Lembre-se de que a formula $P(z_t | x_t) = \alpha \prod_{j}^M{e^{\frac{-(z_j - \hat{z_j})}{2\sigma^2}}}$
        responde somente P(Hi|D), em que H é a hi

        Esta função não precisa retornar nada, mas as partículas precisa ter o seu w recalculado.

        Você vai precisar calcular para o robo

    """ 
    leitura_robo = inspercles.nb_lidar(robot, angles)
    
    for particula in particulas:
        leitura_particula = inspercles.nb_lidar(particula,angles)
        prod = 1
        for angulo in leitura_particula.keys():
            leit_robo = leitura_robo[angulo]
            leit_part = leitura_particula[angulo]
            prod *= np.exp(-(leit_part - leit_robo)/2*(1.5**2))
        particula.w = prod
    
    #Normaliza a nuvem de particulas 
    inspercles.normalize_particles(particulas)
    
    #normalize_particles(particle_cloud) faz a normalização izi pizzi     
    # Voce vai precisar calcular a leitura para cada particula usando inspercles.nb_lidar e depois atualizar as probabilidades




def reamostrar(particulas, n_particulas = num_particulas):
    """
        Reamostra as partículas devolvendo novas particulas sorteadas
        de acordo com a probabilidade e deslocadas de acordo com uma variação normal

        O notebook como_sortear tem dicas que podem ser úteis

        Depois de reamostradas todas as partículas precisam novamente ser deixadas com probabilidade igual

        Use 1/n ou 1, não importa desde que seja a mesma
    """
    particulas_pesos = []
    
    for particula in particulas:
            particulas_pesos.append(particula.w)
            particulas_pesos
    
    
    particulas = draw_random_sample(particulas,particulas_pesos,n_particulas)
    
    
    for particula in particulas:
        particula.w = 1
    
    return particulas
