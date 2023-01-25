import os
import json 
import sys

FOLDER = os.path.dirname(os.path.realpath(__file__))

def main():
	GENERATION_files = get_GENERATION_files()
	for i in range(0,500) :
		try:
			fichier = open(FOLDER+"/GENERATION"+str(i)+".json", 'r')
		except Exception as e:
			print(e)
			sys.exit()
		JSONFED = json.load(fichier)
		fichier.close()
		BEST_APPLES_EATEN = getBestApplesEaten(JSONFED)
		NBR_BEST_APPLES_EATEN = getNumberofBestApplesEaten(JSONFED, BEST_APPLES_EATEN)
		
		BEST_PERFORMANCE = getBestPerformance(JSONFED)
		BEST_STEPS = getBestSteps(JSONFED)

		MEDIAN_PERFORMANCE = getMedianPerformance(JSONFED)
		MEDIAN_STEPS = getMedianSteps(JSONFED)
		

		OUTPUT_FILE = open(FOLDER + "/stats_snakes.txt","a")
		OUTPUT_FILE.write("GENERATION "+str(i)+"\n MAX APPLES EATEN : "+str(BEST_APPLES_EATEN)+" ("+str(NBR_BEST_APPLES_EATEN[0])+")\n MAX APPLES EATEN - 1 : "+str(BEST_APPLES_EATEN-1)+" ("+str(NBR_BEST_APPLES_EATEN[1])+")\n MAX APPLES EATEN - 2 : "+str(BEST_APPLES_EATEN-2)+" ("+str(NBR_BEST_APPLES_EATEN[2])+")\n BEST PERFORMANCE : "+str(BEST_PERFORMANCE)+" (MOY : "+str(MEDIAN_PERFORMANCE)+")\n BEST STEPS : "+str(BEST_STEPS)+" (MOY : "+str(MEDIAN_STEPS)+")\n\n")
		OUTPUT_FILE.close()

def getMedianPerformance(JSONFED):
	SNAKES = JSONFED["snakes"]
	total_performance = 0
	for snake in SNAKES :
		total_performance = total_performance + snake["performance"]
	return total_performance/500

def getMedianSteps(JSONFED):
	SNAKES = JSONFED["snakes"]
	total_steps = 0
	for snake in SNAKES :
		total_steps = total_steps + snake["steps"]
	return total_steps/500


def getBestApplesEaten(JSONFED) :
	SNAKES = JSONFED["snakes"]
	best = 0
	for snake in SNAKES :
		if snake["score"]-2 > best :
			best = snake["score"]-2
	return best

# Retourne une liste comportant le nombre de snakes avec le plus de pommes mangées, puis best -1 puis best -2
def getNumberofBestApplesEaten(JSONFED, best) :
	SNAKES = JSONFED["snakes"]
	compt_best = 1
	compt_best_1 = 0
	compt_best_2 = 0
	for snake in SNAKES :
		if snake["score"]-2 == best :
			compt_best = compt_best + 1
		elif snake["score"]-2 == best-1 :
			compt_best_1 = compt_best_1 + 1
		elif snake["score"]-2 == best-2 :
			compt_best_2 = compt_best_2 + 1
	return [compt_best, compt_best_1, compt_best_2]

def getBestPerformance(JSONFED) :
	SNAKES = JSONFED["snakes"]
	best = 0
	for snake in SNAKES :
		if snake["performance"] > best :
			best = snake["performance"]
	return best

def getBestSteps(JSONFED) :
	SNAKES = JSONFED["snakes"]
	best = 0
	for snake in SNAKES :
		if snake["steps"] > best :
			best = snake["steps"]
	return best

def get_GENERATION_files():
	files = []
	for filename in os.listdir(FOLDER) :
		if filename.find("GENERATION") != -1 and filename.find(".json") != -1 :
			files.append(filename)
	return files

if __name__ == '__main__':
	main()