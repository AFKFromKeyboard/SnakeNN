# SnakeAI
Python script that runs a Snake game and tries to resolve it through NN &amp; genetic algorithm

To start from 0 :

    > python3 snake_ia_no_interface.py

To start at generation 10 :

    > python3 snake_ia_no_interface.py 10
    
Each generation generates a number of snakes (300, 500, etc) and saves the 10 best to be replicated for the next generation.
These parameters can be changed at the top of the script.

To play with a specific snake :

    > python focus_snake.py <id_generation> <id_snake>
    > python focus_snake.py 10 234          #will play with snake 234 from the 10th generation


To generate statistics about the generations (best snake, average score, eaten apples, ...) :

    > python analyze_GENERATIONS.py
