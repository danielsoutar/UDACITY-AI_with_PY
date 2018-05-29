def create_cast_list(filename):
    cast_list = []
    with open(filename, 'r') as file:
        for line in file:
            cast_list.append(line[:line.find(",")])
    return cast_list

cast_list = create_cast_list('flying_circus_cast.txt')
for actor in cast_list:
    print(actor)
