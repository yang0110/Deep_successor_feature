from maze_room import GridMazeDomain
def fourrooms_2(reward_location=[24], init_state=None):
    height = 13
    width = 13
    reward_location = reward_location
    initial_state = init_state  # np.array([25])
    obstacles_location = [42, 43,  49, 50, 120, 121, 140, 141]  # range(height*width)
    v_walls_location = []
    h_walls_location = []
    h_walls_location.extend(range(78, 80))
    h_walls_location.extend(range(81, 85))
    h_walls_location.extend(range(98, 100))
    h_walls_location.extend(range(101, 104))
    v_walls_location.extend(range(6, 45, 13))
    v_walls_location.extend(range(58, 124, 13))
    v_walls_location.extend(range(149, 163, 13))
    walls_location = (h_walls_location, v_walls_location)
    obstacles_transition_probability = .2
    domain = GridMazeDomain(height, width, reward_location,
                                         walls_location, obstacles_location, initial_state,
                                        obstacles_transition_probability, inner_walls=False)
    return domain, reward_location, walls_location[0]+walls_location[1], obstacles_location, height, width
