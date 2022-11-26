# create a virtualenv and activate it
# pip install maturin numpy

# maturin develop --features python
# (from inside the git repo and activated virtual env)

from rubikscube import Cube, HalfTurnMetric, QuarterTurnMetric

cube = Cube.cube_qtm()
cube = Cube.cube_htm()

cube = Cube(HalfTurnMetric)

print("The text repr of the cube")
print(cube)  # text repr of cube

print("\nThe numpy repr of the cube")
print(cube.representation())  # vector repr of cube state
print(cube.representation().shape)  # vector repr of cube state

print("All possible turns, they correspond to 0-11 for qtm and 0-17 for htm")
t = cube.all_possible_turns()
print(cube.all_possible_turns())  # the action space

print("Performing a L turn")
cube.turn(0)  # a L turn

print("Saving cube state")
print(cube)
state = cube.get_state()  # save cube state

print("Scrambling 1000 times")
cube.scramble(1000)  # perform 1000 random moves
print(cube)

print("Restoring cube from before scramble")
cube.set_state(state)
print(cube)  # same as the saved state
