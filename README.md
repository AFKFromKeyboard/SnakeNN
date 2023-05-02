# SnakeAI
Python script that runs a Snake game and tries to resolve it through NN &amp; genetic algorithm

## Play 

### To start from generation 0 (first occurrence):

`> python3 snake_ia_no_interface.py`

### To start at generation N :   
*#requires to have generated the (N-1)th generation in the same folder (GENERATION[N-1].json)*

`> python3 snake_ia_no_interface.py N`
    
**Example:** Generating from the 10th generation from the GENERATION9.json:

`> python3 snake_ia_no_interface.py 10`
    
### To play with a specific snake :

`> python focus_snake.py <id_generation> <id_snake>`
`> python focus_snake.py 10 234`          #will play with snake 234 from the 10th generation
    
Each generation generates a number of snakes (300, 500, etc) and selects the 10 best to be replicated for the next generation.
These parameters can be changed at the top of the script.

Mutations can occur whenever during the generation of a snake: every new weight has a change to mutate. The mutation parameter is defined at the top of the script and can be modified. 

## Statistics
To generate statistics about the generations (best snake, average score, eaten apples, ...) :

`> python analyze_GENERATIONS.py`
