#coding:utf-8

from tkinter import *
import tkinter
import time
import sys
from random import *
import numpy as np
import math
import os
import json

dir_path_last_GEN = "C:\\"
dir_path = os.path.dirname(os.path.realpath(__file__))
LONGUEUR_FENETRE = 700						# Longueur de la fenêtre
HAUTEUR_FENETRE = 700						# Hauteur de la fenêtre
DIMENSIONS_BOARD = 10						# Dimensions du jeu (ex : 21 = 21x21)
DIRECTION = "DOWN"							# Direction du snake
DELAY = 70 # pause before a snake move (ms)	# Délai entre chaque mouvement du snake
LAST_HEAD = "0x0"							# Coordonnées de la dernière tête du snake
LAST_QUEUE = "0x0"							# Coordonnées de la dernière queue du snake
ACTUAL_NEURAL = ""							# Réseau de neurones actuel
SNAKE_ID = 0								# Numéro du snake dans la génération
LOG_COUNT = 0								# Compteur avant d'enregistrer les snakes de LOG_SNAKES dans le fichier de logs
LOG_SNAKES = []								# Liste qui stocke les snakes en cours avant qu'ils soient enregistrés dans le fichier de logs
STEPS = 0									# Nombre de pas du serpent en cours
GENERATION = 0								# Numéro de la génération en cours
NBR_SNAKES_TO_REPRODUCE = 7					# Nombre de snakes à reproduire (= taille de la liste BEST_SNAKES)
NBR_SNAKES_IN_GENERATION = 500				# Nombre de snakes dans une génération
BEST_SNAKES = []							# Liste des meilleurs snakes de la génération précédente
MUTATION_RATE = 0							# Pourcentage de mutation à chaque création d'un noeud pour un nouveau snake (à partir de la deuxième GENERATION)
COMPT_BEFORE_SAVE_SNAKES = 20				# Compteur avant sauvegarde des snakes dans le JSON
MAX_STEPS = 100

args = sys.argv								# Pour commencer d'une génération donnée
if len(args) != 3 :
	print("2 arguments needed !")
	print("py focus_snake.py [GENERATION_ID] [SNAKE_ID]")
	sys.exit()
else:
	GENERATION = int(args[1])
	SNAKE_ID_ARGS = int(args[2])


# CLASSE NEURONE
# Chaque ligne de matrice contient les poids liés à 1 node
# Le nombre de colonnes est le nombre de neurones contenus dans le layer
class Neural(object):		
	Wb = np.array([])		# Wb matrice de 6x16 (6 lignes, 16 colonnes)
	Wc = np.array([])		# Wc matrice de 6x6 (6 lignes, 6 colonnes)
	Wd = np.array([])		# Wd matrice de 4x6 (4 lignes, 6 colonnes)
	A = np.array([])		# A matrice de 16
	B = np.array([])		# B matrice de 6
	C = np.array([])		# C matrice de 6
	D = np.array([])		# D matrice de 4

	def __init__(self, Wb=[], Wc=[], Wd=[], A=[], B=[], C=[], D=[], random_inputs=False):
		self.A = A
		self.B = B
		self.C = C
		self.D = D
		self.Wb = Wb
		self.Wc = Wc
		self.Wd = Wd 
		if random_inputs:
			self.Wb = np.random.uniform(low=-1.0,high=1.0,size=(6,16))
			self.Wc = np.random.uniform(low=-1.0,high=1.0,size=(6,6))
			self.Wd = np.random.uniform(low=-1.0,high=1.0,size=(4,6))
		else :
			poids = get_snake_poids()
			self.Wb = poids["Wb"]
			self.Wc = poids["Wc"]
			self.Wd = poids["Wd"]

	def calculate_A(self):
		global snake
		distance = get_distance_to_wall(snake)
		A = np.array([
			distance[0]**(-1),									# Distance au mur du haut (réduit dans un intervalle [0,1])
			distance[1]**(-1),									# Distance au mur de gauche (réduit dans un intervalle [0,1])
			distance[2]**(-1),									# Distance au mur de droite (réduit dans un intervalle [0,1])
			distance[3]**(-1),									# Distance au mur du bas (réduit dans un intervalle [0,1])
			snake.is_there_apple_on_line_top(),					# Il y a une pomme vers le haut ? Oui : 1 ; Non : 0
			snake.is_there_apple_on_line_left(),				# Il y a une pomme vers la gauche ? Oui : 1 ; Non : 0
			snake.is_there_apple_on_line_right(),				# Il y a une pomme vers la droite ? Oui : 1 ; Non : 0
			snake.is_there_apple_on_line_bot(),					# Il y a une pomme vers le bas ? Oui : 1 ; Non : 0
			snake.is_there_snake_queue_on_top(),				# Il y a la queue du serpent vers le haut ? Oui : 1 ; Non : 0
			snake.is_there_snake_queue_on_left(),				# Il y a la queue du serpent vers la gauche ? Oui : 1 ; Non : 0
			snake.is_there_snake_queue_on_right(),				# Il y a la queue du serpent vers la droite ? Oui : 1 ; Non : 0
			snake.is_there_snake_queue_on_bot(),				# Il y a la queue du serpent vers le bas ? Oui : 1 ; Non : 0
			snake.is_there_apple_on_diagonale_HautGauche(),		# Il y a une pomme sur la diagonale haut-gauche ? Oui : 1 ; Non : 0
			snake.is_there_apple_on_diagonale_HautDroite(),		# Il y a une pomme sur la diagonale haut-droite ? Oui : 1 ; Non : 0
			snake.is_there_apple_on_diagonale_BasGauche(),		# Il y a une pomme sur la diagonale bas-gauche ? Oui : 1 ; Non : 0
			snake.is_there_apple_on_diagonale_BasDroite()		# Il y a une pomme sur la diagonale bas-droite ? Oui : 1 ; Non : 0
			])
		for i in range(0,4) :
			A[i] = 1 - A[i] 
		return A

  	# Calcule la matrice B
	def calculate_B(self,A,Wb):
		INDEX1 = tanh(np.dot(Wb[0],A))
		INDEX2 = tanh(np.dot(Wb[1],A))
		INDEX3 = tanh(np.dot(Wb[2],A))
		INDEX4 = tanh(np.dot(Wb[3],A))
		INDEX5 = tanh(np.dot(Wb[4],A))
		INDEX6 = tanh(np.dot(Wb[5],A))
		B = np.array([INDEX1,INDEX2,INDEX3,INDEX4,INDEX5,INDEX6])
		return(B)

  	# Calcule la matrice C
	def calculate_C(self,B,Wc):
		INDEX1 = tanh(np.dot(Wc[0],B))
		INDEX2 = tanh(np.dot(Wc[1],B))
		INDEX3 = tanh(np.dot(Wc[2],B))
		INDEX4 = tanh(np.dot(Wc[3],B))
		INDEX5 = tanh(np.dot(Wc[4],B))
		INDEX6 = tanh(np.dot(Wc[5],B))
		C = np.array([INDEX1,INDEX2,INDEX3,INDEX4,INDEX5,INDEX6])
		return(C)

  	# Calcule la matrice D
	def calculate_D(self,C,Wd):
		INDEX1 = tanh(np.dot(Wd[0],C))
		INDEX2 = tanh(np.dot(Wd[1],C))
		INDEX3 = tanh(np.dot(Wd[2],C))
		INDEX4 = tanh(np.dot(Wd[3],C))
		D = np.array([INDEX1,INDEX2,INDEX3,INDEX4])
		return(D)

# CLASSE BOARD/PLATEAU
class Board(object):
	DIMENSIONS_BOARD = 10
	CANVAS = "Objet Canvas"
	COORDS_NUMERO = {}
	COORDS_PLACEMENT = {}
	COORDS_APPLE = "0x0"

	# Initialise un Board
	# INPUT : dimension du Board (INTEGER)
	def __init__(self, DIMENSIONS_BOARD):
		self.DIMENSIONS_BOARD = DIMENSIONS_BOARD
		COORDS_NUMERO = {}
		COORDS_PLACEMENT = {}
		longueur_canvas = LONGUEUR_FENETRE*0.8
		hauteur_canvas = HAUTEUR_FENETRE*0.8
		canvas = Canvas(fenetre, width=longueur_canvas, height=hauteur_canvas, background='white')
		k = 1 # Compteur pour numéroter les rectangles (de 1 à n*n) avec n la dimension du Board
		for i in range (0,DIMENSIONS_BOARD) : # COORDS Y
			for j in range (0,DIMENSIONS_BOARD) :	# COORDS X
				canvas.create_rectangle(j*(longueur_canvas/DIMENSIONS_BOARD), i*((hauteur_canvas)/DIMENSIONS_BOARD), (j+1)*(longueur_canvas/DIMENSIONS_BOARD), (i+1)*((hauteur_canvas)/DIMENSIONS_BOARD),width=0)
				COORDS_NUMERO[k] = {"X0":j*(longueur_canvas/DIMENSIONS_BOARD), "Y0":i*((hauteur_canvas)/DIMENSIONS_BOARD), "X1":(j+1)*(longueur_canvas/DIMENSIONS_BOARD), "Y1":(i+1)*((hauteur_canvas)/DIMENSIONS_BOARD)}
				COORDS_PLACEMENT[str(j)+"x"+str(i)] = {"X0":j*(longueur_canvas/DIMENSIONS_BOARD), "Y0":i*((hauteur_canvas)/DIMENSIONS_BOARD), "X1":(j+1)*(longueur_canvas/DIMENSIONS_BOARD), "Y1":(i+1)*((hauteur_canvas)/DIMENSIONS_BOARD)}
				k = k + 1
		self.CANVAS = canvas # Objet Canvas instancié, contient toutes les cases du plateau numérotées de 1 à dimension*dimension
		self.COORDS_NUMERO = COORDS_NUMERO # Dictionnaire retournant pour chaque case ses coordonnées (x0,y0,x1,y1), en fonction de son numéro
		self.COORDS_PLACEMENT = COORDS_PLACEMENT # Dictionnaire retournant pour chaque case ses coordonnées (x0,y0,x1,y1) en fonction de son placement (1x2, 3x3, 5x2, ...)


	# Retourne un dictionnaire contenant les coordonnées d'une case (X0,Y0,X1,Y1)
	# INPUT : placement de la case sur le plateau (par exemple 1x1, nxn) (STRING)
	def get_coords_placement(self, placement):
		return self.COORDS_PLACEMENT[placement]

	# Passe la couleur de la case à rouge
	# INPUT : placement de la case sur le plateau (par exemple de 1x1 à nxn) (STRING)
	def set_color_red(self, placement):
		COORDS = self.get_coords_placement(placement)
		self.CANVAS.create_rectangle(COORDS["X0"],COORDS["Y0"],COORDS["X1"],COORDS["Y1"],fill="red",width=1,outline="black")

	# Passe la couleur de la case à blanc
	# INPUT : placement de la case sur le plateau (par exemple de 1x1 à nxn) (STRING)
	def set_color_white(self, placement):
		COORDS = self.get_coords_placement(placement)
		self.CANVAS.create_rectangle(COORDS["X0"],COORDS["Y0"],COORDS["X1"],COORDS["Y1"],fill="white",width=1,outline="white")

	# Passe la couleur de la case à noir
	# INPUT : placement de la case sur le plateau (par exemple de 1x1 à nxn) (STRING)
	def set_color_black(self, placement):
		COORDS = self.get_coords_placement(placement)
		self.CANVAS.create_rectangle(COORDS["X0"],COORDS["Y0"],COORDS["X1"],COORDS["Y1"],fill="black",width=1,outline="black")

	# Passe la couleur de la case à vert
	# INPUT : placement de la case sur le plateau (par exemple de 1x1 à nxn) (STRING)
	def set_color_green(self, placement):
		COORDS = self.get_coords_placement(placement)
		self.CANVAS.create_rectangle(COORDS["X0"],COORDS["Y0"],COORDS["X1"],COORDS["Y1"],fill="green",width=1,outline="black")

	# Remet toutes les cases du board en blanc
	def set_all_white(self):
		cases = get_ALL_placements(self)
		for case in cases :
			self.set_color_white(case)

	# Actualise les cases avec les couleurs qui vont bien (en fonction de la position de la pomme, serpent, etc)
	def change_board_colors(self):
		self.set_color_white(LAST_QUEUE)
		self.set_color_green(self.COORDS_APPLE)
		self.set_color_red(snake.COORDS_HEAD)
		self.set_color_black(LAST_HEAD)
		self.set_color_black(snake.COORDS_QUEUE[-1])

	# Définit les cases avec les couleurs qui vont bien (en fonction de la position de la pomme, serpent, etc)
	def set_board_colors(self):
		self.set_color_white(LAST_QUEUE)
		self.set_color_green(self.COORDS_APPLE)
		self.set_color_red(snake.COORDS_HEAD)
		self.set_color_black(snake.COORDS_QUEUE[-1])
		
	# Redéfinit les couleurs des cases du board à blanc
	def reset(self):
		self.set_all_white()

	# Génère une nouvelle pomme en fonction de la position du serpent
	def generate_apple_case(self):
		global APPLE_COORDS
		ALL_cases = get_ALL_placements(self)
		ALL_cases_but_no_snake = ALL_cases
		ALL_cases.remove(snake.COORDS_HEAD)
		for coords in snake.COORDS_QUEUE :
			try:
				ALL_cases_but_no_snake.remove(coords)
			except:
				pass
		result = choice(ALL_cases_but_no_snake)
		self.COORDS_APPLE = result
		APPLE_COORDS = result

# CLASSE SNAKE
class Snake(object):
	COORDS_HEAD = "1x1"
	COORDS_QUEUE = []
	LENGTH = 2

	def __init__(self):
		global DIMENSIONS_BOARD
		self.LENGTH = 2
		self.COORDS_HEAD = str(randint((DIMENSIONS_BOARD/2)-1,(DIMENSIONS_BOARD/2)))+"x"+str(randint((DIMENSIONS_BOARD/2)-1,(DIMENSIONS_BOARD/2)))
		self.COORDS_QUEUE = []
		self.COORDS_QUEUE.append(str(get_XYplacement(self.COORDS_HEAD)[0]-1)+"x"+str(get_XYplacement(self.COORDS_HEAD)[1]-1))

	# Modifie les coordonnées du serpent pour le déplacer vers le haut
	def moveHead_UP(self):
		old_COORDS_HEAD = self.COORDS_HEAD
		self.COORDS_HEAD = str(get_XYplacement(self.COORDS_HEAD)[0]) + "x" + str(get_XYplacement(self.COORDS_HEAD)[1] - 1)
		if self.LENGTH != 1 :
			new_COORDS_QUEUE = []
			new_COORDS_QUEUE.append(old_COORDS_HEAD)
			for i in range (0, self.LENGTH-2):
				new_COORDS_QUEUE.append(self.COORDS_QUEUE[i])
			self.COORDS_QUEUE = new_COORDS_QUEUE

	# Modifie les coordonnées du serpent pour le déplacer vers le bas
	def moveHead_DOWN(self):
		old_COORDS_HEAD = self.COORDS_HEAD
		self.COORDS_HEAD = str(get_XYplacement(self.COORDS_HEAD)[0]) + "x" + str(get_XYplacement(self.COORDS_HEAD)[1] + 1)
		if self.LENGTH != 1 :
			new_COORDS_QUEUE = []
			new_COORDS_QUEUE.append(old_COORDS_HEAD)
			for i in range (0, self.LENGTH-2):
				new_COORDS_QUEUE.append(self.COORDS_QUEUE[i])
			self.COORDS_QUEUE = new_COORDS_QUEUE
		

	# Modifie les coordonnées du serpent pour le déplacer vers la gauche
	def moveHead_LEFT(self):
		old_COORDS_HEAD = self.COORDS_HEAD
		self.COORDS_HEAD = str(get_XYplacement(self.COORDS_HEAD)[0] - 1) + "x" + str(get_XYplacement(self.COORDS_HEAD)[1])
		if self.LENGTH != 1 :
			new_COORDS_QUEUE = []
			new_COORDS_QUEUE.append(old_COORDS_HEAD)
			for i in range (0, self.LENGTH-2):
				new_COORDS_QUEUE.append(self.COORDS_QUEUE[i])
			self.COORDS_QUEUE = new_COORDS_QUEUE

	# Modifie les coordonnées du serpent pour le déplacer vers la droite
	def moveHead_RIGHT(self):
		old_COORDS_HEAD = self.COORDS_HEAD
		self.COORDS_HEAD = str(get_XYplacement(self.COORDS_HEAD)[0] + 1) + "x" + str(get_XYplacement(self.COORDS_HEAD)[1])
		if self.LENGTH != 1 :
			new_COORDS_QUEUE = []
			new_COORDS_QUEUE.append(old_COORDS_HEAD)
			for i in range (0, self.LENGTH-2):
				new_COORDS_QUEUE.append(self.COORDS_QUEUE[i])
			self.COORDS_QUEUE = new_COORDS_QUEUE

	# Déplace le serpent en fonction de la touche entrée
	def move(self):
		global DIRECTION, LAST_QUEUE, LAST_HEAD
		LAST_HEAD = self.COORDS_HEAD
		LAST_QUEUE = self.COORDS_QUEUE[-1]
		if DIRECTION == "UP":
			self.moveHead_UP()
		elif DIRECTION == "DOWN":
			self.moveHead_DOWN()
		elif DIRECTION == "LEFT":
			self.moveHead_LEFT()
		elif DIRECTION == "RIGHT":
			self.moveHead_RIGHT()
		

	# Ajoute une case au serpent
	def add_case_to_queue(self, case):
		self.COORDS_QUEUE.append(case)

	# Retourne 1 s'il y a la queue du serpent sur la ligne au dessus
	def is_there_snake_queue_on_top(self):
		coords_tete = get_XYplacement(self.COORDS_HEAD)
		for queue in self.COORDS_QUEUE:
			if get_XYplacement(queue)[1] < coords_tete[1]:
				return 1
		return 0

	# Retourne 1 s'il y a la queue du serpent sur la ligne à gauche
	def is_there_snake_queue_on_left(self):
		coords_tete = get_XYplacement(self.COORDS_HEAD)
		for queue in self.COORDS_QUEUE:
			if get_XYplacement(queue)[0] < coords_tete[0]:
				return 1
		return 0

	# Retourne 1 s'il y a la queue du serpent sur la ligne à droite
	def is_there_snake_queue_on_right(self):
		coords_tete = get_XYplacement(self.COORDS_HEAD)
		for queue in self.COORDS_QUEUE:
			if get_XYplacement(queue)[0] > coords_tete[0]:
				return 1
		return 0

	# Retourne 1 s'il y a la queue du serpent sur la ligne en dessous
	def is_there_snake_queue_on_bot(self):
		coords_tete = get_XYplacement(self.COORDS_HEAD)
		for queue in self.COORDS_QUEUE:
			if get_XYplacement(queue)[1] > coords_tete[1]:
				return 1
		return 0

	# Retourne 1 s'il y a un obstacle au dessus de la tête du serpent (mur ou lui-même)
	def is_there_something_on_top(self):
		placement_tete = get_XYplacement(self.COORDS_HEAD)
		if (placement_tete[1] == 0) or (str(placement_tete[0])+"x"+str(placement_tete[1]-1) in self.COORDS_QUEUE):
			return 1
		else :
			return 0

	# Retourne 1 s'il y a un obstacle sur la gauche de la tête du serpent (mur ou lui-même)
	def is_there_something_on_left(self):
		placement_tete = get_XYplacement(self.COORDS_HEAD)
		if (placement_tete[0] == 0) or (str(placement_tete[0]-1)+"x"+str(placement_tete[1]) in self.COORDS_QUEUE):
			return 1
		else :
			return 0

	# Retourne 1 s'il y a un obstacle sur la droite de la tête du serpent (mur ou lui-même)
	def is_there_something_on_right(self):
		placement_tete = get_XYplacement(self.COORDS_HEAD)
		if (placement_tete[0] == DIMENSIONS_BOARD-1) or (str(placement_tete[0]+1)+"x"+str(placement_tete[1]) in self.COORDS_QUEUE):
			return 1
		else :
			return 0

	# Retourne 1 s'il y a un obstacle en dessous de la tête du serpent (mur ou lui-même)
	def is_there_something_on_bot(self):
		placement_tete = get_XYplacement(self.COORDS_HEAD)
		if (placement_tete[1] == DIMENSIONS_BOARD-1) or (str(placement_tete[0])+"x"+str(placement_tete[1]+1) in self.COORDS_QUEUE):
			return 1
		else :
			return 0

	# Retourne 1 si une pomme est sur la même colonne que la tête, vers le haut
	def is_there_apple_on_line_top(self):
		coords_apple = get_XYplacement(board.COORDS_APPLE)
		placement_tete = get_XYplacement(self.COORDS_HEAD)
		if coords_apple[0] == placement_tete[0] and coords_apple[1] < placement_tete[1]:
			return 1
		else:
			return 0

	# Retourne 1 si une pomme est sur la même ligne que la tête, vers la gauche
	def is_there_apple_on_line_left(self):
		coords_apple = get_XYplacement(board.COORDS_APPLE)
		placement_tete = get_XYplacement(self.COORDS_HEAD)
		if coords_apple[0] < placement_tete[0] and coords_apple[1] == placement_tete[1]:
			return 1
		else:
			return 0

	# Retourne 1 si une pomme est sur la même ligne que la tête, vers la droite
	def is_there_apple_on_line_right(self):
		coords_apple = get_XYplacement(board.COORDS_APPLE)
		placement_tete = get_XYplacement(self.COORDS_HEAD)
		if coords_apple[0] > placement_tete[0] and coords_apple[1] == placement_tete[1]:
			return 1
		else:
			return 0

	# Retourne 1 si une pomme est sur la même colonne que la tête, vers le bas
	def is_there_apple_on_line_bot(self):
		coords_apple = get_XYplacement(board.COORDS_APPLE)
		placement_tete = get_XYplacement(self.COORDS_HEAD)
		if coords_apple[0] == placement_tete[0] and coords_apple[1] > placement_tete[1]:
			return 1
		else:
			return 0

	# Retourne 1 si une pomme est sur la diagonale HautGauche de la tête
	def is_there_apple_on_diagonale_HautGauche(self):
		coords_apple = board.COORDS_APPLE
		diagonale = self.get_diagonale_HautGauche()
		for case in diagonale :
			if coords_apple == case :
				return 1
		return 0

	# Retourne 1 si une pomme est sur la diagonale HautDroite de la tête
	def is_there_apple_on_diagonale_HautDroite(self):
		coords_apple = board.COORDS_APPLE
		diagonale = self.get_diagonale_HautDroite()
		for case in diagonale :
			if coords_apple == case :
				return 1
		return 0

	# Retourne 1 si une pomme est sur la diagonale BasGauche de la tête
	def is_there_apple_on_diagonale_BasGauche(self):
		coords_apple = board.COORDS_APPLE
		diagonale = self.get_diagonale_BasGauche()
		for case in diagonale :
			if coords_apple == case :
				return 1
		return 0

	# Retourne 1 si une pomme est sur la diagonale BasDroite de la tête
	def is_there_apple_on_diagonale_BasDroite(self):
		coords_apple = board.COORDS_APPLE
		diagonale = self.get_diagonale_BasDroite()
		for case in diagonale :
			if coords_apple == case :
				return 1
		return 0

	# Retourne un dictionnaire contenant les coordonnées de chaque diagonale (sous la forme de tableaux)
	def get_snake_diagonales_coords(self):
		return {'Diagonale_HautGauche': self.get_diagonale_HautGauche(), 'Diagonale_HautDroite': self.get_diagonale_HautDroite(), 'Diagonale_BasGauche': self.get_diagonale_BasGauche(), 'Diagonale_BasDroite': self.get_diagonale_BasDroite()}

	# Retourne une liste des coordonnées de la diagonale HautGauche à la tête
	def get_diagonale_HautGauche(self):
		placement_tete = get_XYplacement(self.COORDS_HEAD)
		diagonale = []
		x = placement_tete[0]
		y = placement_tete[1]
		while x != 0 and y != 0 :
			x = x - 1
			y = y - 1
			diagonale.append(str(x)+"x"+str(y))
		return diagonale

	# Retourne une liste des coordonnées de la diagonale HautDroite à la tête
	def get_diagonale_HautDroite(self):
		placement_tete = get_XYplacement(self.COORDS_HEAD)
		diagonale = []
		x = placement_tete[0]
		y = placement_tete[1]
		while x != DIMENSIONS_BOARD-1 and y != 0 :
			x = x + 1
			y = y - 1
			diagonale.append(str(x)+"x"+str(y))
		return diagonale

	# Retourne une liste des coordonnées de la diagonale BasGauche à la tête
	def get_diagonale_BasGauche(self):
		placement_tete = get_XYplacement(self.COORDS_HEAD)
		diagonale = []
		x = placement_tete[0]
		y = placement_tete[1]
		while x != 0 and y != DIMENSIONS_BOARD-1 :
			x = x - 1
			y = y + 1
			diagonale.append(str(x)+"x"+str(y))
		return diagonale

	# Retourne une liste des coordonnées de la diagonale BasDroite à la tête
	def get_diagonale_BasDroite(self):
		placement_tete = get_XYplacement(self.COORDS_HEAD)
		diagonale = []
		x = placement_tete[0]
		y = placement_tete[1]
		while x != DIMENSIONS_BOARD-1 and y != DIMENSIONS_BOARD-1 :
			x = x + 1
			y = y + 1
			diagonale.append(str(x)+"x"+str(y))
		return diagonale

	# Retourne un INT (indice de performance des snakes)
	def calculate_performance(self, steps):
		if steps == MAX_STEPS :
			return 0
		else :
			eaten_apples = self.LENGTH-2
			return int((steps + (2**eaten_apples) + (450*(eaten_apples**2.1)) - (0.4*(steps**1.3)*(eaten_apples**1.2))))
			#return(((self.LENGTH)**(self.LENGTH-1))+((self.LENGTH-1)*steps)) 					ORIGINAL PERFORMANCE FUNCTION

# Retourne la moyenne de deux matrice
# OUTPUT : matrice
def get_mean_matrix(A, B):
	mean = A.__add__(B)
	return mean/2

# Retourne une matrice Wb à partir d'un élément dans le json
def str2array(Wb):
	Wb = Wb.replace("[[","[")
	Wb = Wb.replace("]]","]")

	mat1 = Wb.split("] ")

	mat1 = [s.strip('[') for s in mat1]
	mat1 = [s.strip(']') for s in mat1]

	array = np.fromstring(mat1[0],dtype=float,sep=" ")
	for i in range(1,len(mat1)) :
		array = np.vstack((array,np.fromstring(mat1[i],dtype=float,sep=" ")))

	return array

# Retourne True ou False pour désigner s'il va y avoir une mutation
# mutation_rate est le pourcentage de chance d'avoir une mutation
def is_there_mutation(mutation_rate):
	r = uniform(0, 100)
	if r <= float(mutation_rate) :
		return True
	else :
		return False

# Retourne des matrices de poids générées avec les meilleurs snakes
# OUTPUT : dictionnaire de trois matrices {"Wb": WB, "Wc": WB, "Wc": WC}
def get_snake_poids():
	snake = get_snake()
	WB_list = []
	sublist = []
	for i in range(0,6) :
		for j in range(0,16) :
			if is_there_mutation(MUTATION_RATE) :
				sublist.append(uniform(-1.0,1.0))
			else :
				try:
					sublist.append(str2array(snake["Wb"])[i][j])
				except:
					print("ERROR WHILE SELECTING NEW Wb")
					
		WB_list.append(sublist)
		sublist = []
	WB = np.array(WB_list)
	#WC = str2array(BEST_SNAKES[a]["Wc"]),str2array(BEST_SNAKES[b]["Wc"])
	#WD = str2array(BEST_SNAKES[a]["Wd"]),str2array(BEST_SNAKES[b]["Wd"])
	WC_list = []
	sublist2 = []
	for k in range(0,6) :
		for l in range(0,6) :
			if is_there_mutation(MUTATION_RATE) :
				sublist2.append(uniform(-1.0,1.0))
			else :
				try:
					sublist2.append(str2array(snake["Wc"])[k][l])
				except:
					print("ERROR WHILE SELECTING NEW Wc")
					
		WC_list.append(sublist2)
		sublist2 = []
	WC = np.array(WC_list)

	WD_list = []
	sublist3 = []
	for m in range(0,4) :
		for n in range(0,6) :
			if is_there_mutation(MUTATION_RATE) :
				sublist3.append(uniform(-1.0,1.0))
			else :
				try:
					sublist3.append(str2array(snake["Wd"])[m][n])
				except:
					print("ERROR WHILE SELECTING NEW Wd")
					
		WD_list.append(sublist3)
		sublist3 = []
	WD = np.array(WD_list)
	return {"Wb": WB, "Wc": WC, "Wd": WD}


# Retourne le snake correspondant à la GENERATION et à L'ID
def get_snake():
	fichier = open(dir_path+"/GENERATION"+str(GENERATION)+".json", 'r')
	jsonn = json.load(fichier)
	fichier.close()
	snakes = jsonn['snakes']
	for snake in snakes :
		if snake["snake_id"] == SNAKE_ID_ARGS :
			return snake
		else :
			pass
	print("Snake with ID "+str(SNAKE_ID_ARGS)+" not found")
	sys.exit()


# Retourne un tableau contenant la distance de la tête aux murs [haut,gauche,droite,bas]
def get_distance_to_wall(snake):
	coords_tete = get_XYplacement(snake.COORDS_HEAD)
	return [coords_tete[1]+1,coords_tete[0]+1,DIMENSIONS_BOARD-coords_tete[0],DIMENSIONS_BOARD-coords_tete[1]]

# Retourne un dictionnaire comportant l'id du snake, son score, sa performance et ses poids
def store_snake_performance(neural,snake_id,snake):
	print("APPLES EATEN : "+str(snake.LENGTH - 2))
	performance = snake.calculate_performance(STEPS)
	print("PERFORMANCE : "+str(performance))
	return {"snake_id":snake_id,"score":snake.LENGTH,"Wb":np.array_str(neural.Wb).replace("\n",""),"Wc":np.array_str(neural.Wc).replace("\n",""),"Wd":np.array_str(neural.Wd).replace("\n",""),"steps":STEPS, "performance":performance}

# Crée un nouveau fichier de sauvegarde des JSON
def create_new_log_file():
	with open(dir_path+"/GENERATION"+str(GENERATION)+"_SNAKE"+str(SNAKE_ID_ARGS)+".json", 'w+',encoding='utf-8') as file:
		file.write('{"snakes": []}')
		file.close()

# Ajoute le snake actuel aux logs
def log_snakes():
	fichier = open(dir_path+"/GENERATION"+str(GENERATION)+"_SNAKE"+str(SNAKE_ID_ARGS)+".json", 'r')
	jsonn = json.load(fichier)
	fichier.close()
	fichier = open(dir_path+"/GENERATION"+str(GENERATION)+"_SNAKE"+str(SNAKE_ID_ARGS)+".json", 'w+')
	for snake in LOG_SNAKES :
		jsonn['snakes'].append(snake)
	json.dump(jsonn,fichier)
	fichier.close()

# Calcule la sigmoide de x
def sigmoid(x):

	return 1 / (1 + math.exp(-x))

# Calcule la tanh de x
def tanh(x):
	try:
		return (math.exp(x) - math.exp(-x)) / (math.exp(x) + math.exp(-x))
	except Exception as e:
		print("--------------------------------------")
		print("----------------DEBUG-----------------")
		print(x)
		raise e


def change_direction_with_neural(neural_network):
	global DIRECTION
	matrice = neural_network.D
	if matrice[0] > matrice[1] and matrice[0] > matrice[2] and matrice[0] > matrice[3]:
		DIRECTION = "UP"
	elif matrice[1] > matrice[0] and matrice[1] > matrice[2] and matrice[1] > matrice[3]:
		DIRECTION = "LEFT"
	elif matrice[2] > matrice[0] and matrice[2] > matrice[1] and matrice[2] > matrice[3]:
		DIRECTION = "RIGHT"
	elif matrice[3] > matrice[0] and matrice[3] > matrice[1] and matrice[3] > matrice[2]:
		DIRECTION = "DOWN"
	else:
		print("Probleme dans change_direction_with_neural")

# Retourne un tableau avec toutes les coordonnées d'un board
def get_ALL_placements(board):
	result = []
	for i in range (0, board.DIMENSIONS_BOARD):
		for j in range (0, board.DIMENSIONS_BOARD):
			result.append(str(j)+"x"+str(i))
	return result

# Modifie la direction à prendre pour le serpent vers la gauche
def left(event):
    global DIRECTION
    DIRECTION = "LEFT"
 
# Modifie la direction à prendre pour le serpent vers la droite
def right(event):
    global DIRECTION
    DIRECTION = "RIGHT"
 
# Modifie la direction à prendre pour le serpent vers le haut
def up(event):
    global DIRECTION
    DIRECTION = "UP"

# Modifie la direction à prendre pour le serpent vers le bas
def down(event):
    global DIRECTION
    DIRECTION = "DOWN"

# Retourne un tableau contenant les coordonnées [X,Y] d'une case en placement, en INTEGER
# INPUT : placement de la case sur le plateau (par exemple de 1x1 à nxn) (STRING)
def get_XYplacement(placement):
	return int(placement.split("x")[0]),int(placement.split("x")[1])

# Réinitialise les paramètres du jeu si fin de partie
def lose():
	global board, snake, DIRECTION, LAST_QUEUE, LAST_HEAD, ACTUAL_NEURAL, SNAKE_ID, LOG_COUNT, STEPS, LOG_SNAKES
	print("\n")
	print("SNAKE_ID : "+str(SNAKE_ID))
	# print("*** ACTUAL_NEURAL.A : \n"+np.array_str(ACTUAL_NEURAL.A))
	# print("-----------------------------\r")
	# print("*** ACTUAL_NEURAL.Wb : \n"+np.array_str(ACTUAL_NEURAL.Wb))
	# print("-----------------------------\r")
	# print("*** ACTUAL_NEURAL.B : \n"+np.array_str(ACTUAL_NEURAL.B))
	# print("-----------------------------\r")
	# print("*** ACTUAL_NEURAL.Wc : \n"+np.array_str(ACTUAL_NEURAL.Wc))
	# print("-----------------------------\r")
	# print("*** ACTUAL_NEURAL.C : \n"+np.array_str(ACTUAL_NEURAL.C))
	# print("-----------------------------\r")
	# print("*** ACTUAL_NEURAL.Wd : \n"+np.array_str(ACTUAL_NEURAL.Wd))
	# print("-----------------------------\r")
	# print("*** ACTUAL_NEURAL.D : \n"+np.array_str(ACTUAL_NEURAL.D))
	#print("You lost!")
	LOG_SNAKES.append(store_snake_performance(ACTUAL_NEURAL,SNAKE_ID,snake))
	if LOG_COUNT == COMPT_BEFORE_SAVE_SNAKES :
		#log_snakes()
		LOG_SNAKES = []
		LOG_COUNT = 0
	else: 
		LOG_COUNT = LOG_COUNT + 1
	STEPS = 0
	SNAKE_ID = SNAKE_ID + 1
	newGame()

# Retourne True si la tête du serpent est sur la pomme
# Retourne False sinon
def snake_is_on_apple():
	global snake, board
	if snake.COORDS_HEAD == board.COORDS_APPLE :
		return True
	else:
		return False

def newGame():
	global board, snake, DIRECTION, LAST_QUEUE, LAST_HEAD, ACTUAL_NEURAL
	print("-----------------------------\n")
	snake=Snake()
	board.reset()
	board.generate_apple_case()
	board.set_board_colors()
	DIRECTION = "DOWN"
	LAST_HEAD = "0x0"
	LAST_QUEUE = "0x0"
	ACTUAL_NEURAL = Neural(random_inputs=False)
	Game()

def Game():
	global board, DIRECTION, snake, fenetre, ACTUAL_NEURAL, STEPS
	
	ACTUAL_NEURAL.A = ACTUAL_NEURAL.calculate_A()
	ACTUAL_NEURAL.B = ACTUAL_NEURAL.calculate_B(ACTUAL_NEURAL.A,ACTUAL_NEURAL.Wb)
	ACTUAL_NEURAL.C = ACTUAL_NEURAL.calculate_C(ACTUAL_NEURAL.B,ACTUAL_NEURAL.Wc)
	ACTUAL_NEURAL.D = ACTUAL_NEURAL.calculate_D(ACTUAL_NEURAL.C,ACTUAL_NEURAL.Wd)
	change_direction_with_neural(ACTUAL_NEURAL)

	snake.move()

	if get_XYplacement(snake.COORDS_HEAD)[0] == -1 or get_XYplacement(snake.COORDS_HEAD)[0] == DIMENSIONS_BOARD or get_XYplacement(snake.COORDS_HEAD)[1] == -1 or get_XYplacement(snake.COORDS_HEAD)[1] == DIMENSIONS_BOARD :
		lose()
		return
	if snake.COORDS_HEAD in snake.COORDS_QUEUE or snake.COORDS_HEAD == LAST_QUEUE or STEPS == MAX_STEPS :
		lose()
		return
	STEPS = STEPS + 1

	# print("*** ACTUAL_NEURAL.A : \n"+np.array_str(ACTUAL_NEURAL.A))
	# print("-----------------------------\r")
	# print("*** ACTUAL_NEURAL.Wb : \n"+np.array_str(ACTUAL_NEURAL.Wb))
	# print("-----------------------------\r")
	# print("*** ACTUAL_NEURAL.B : \n"+np.array_str(ACTUAL_NEURAL.B))
	# print("-----------------------------\r")
	# print("*** ACTUAL_NEURAL.Wc : \n"+np.array_str(ACTUAL_NEURAL.Wc))
	# print("-----------------------------\r")
	# print("*** ACTUAL_NEURAL.C : \n"+np.array_str(ACTUAL_NEURAL.C))
	# print("-----------------------------\r")
	# print("*** ACTUAL_NEURAL.Wd : \n"+np.array_str(ACTUAL_NEURAL.Wd))
	# print("-----------------------------\r")
	# print("*** ACTUAL_NEURAL.D : \n"+np.array_str(ACTUAL_NEURAL.D))
	if snake_is_on_apple() :
		snake.LENGTH = snake.LENGTH + 1
		snake.add_case_to_queue(LAST_QUEUE)
		board.generate_apple_case()
	board.change_board_colors()
	fenetre.after(DELAY, Game)

#create_new_log_file()
snake=Snake()
fenetre = tkinter.Tk()
fenetre.title("Snake game")
fenetre.geometry(str(LONGUEUR_FENETRE)+"x"+str(HAUTEUR_FENETRE))

board = Board(DIMENSIONS_BOARD)
board.generate_apple_case()
board.CANVAS.pack(side=TOP, padx=5, pady=5)


button = tkinter.Button(fenetre, text="Play", width=10, height=2, command=newGame)	
# affiche un bouton sur l'interface, l'attribut "command" appelle une fonction et agit comme un "onclick"
button.pack()

fenetre.bind('<d>', right)
fenetre.bind('<q>', left)
fenetre.bind('<z>' , up)
fenetre.bind('<s>', down)

fenetre.mainloop()				# pour looper la fenêtre et l'afficher
fenetre.quit()					# pour quitter la fenêtre

