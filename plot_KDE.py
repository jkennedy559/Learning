data = (2, 2, 5, 6, 7)


def compute_dimensions(points):
    block_coordinates = map(lambda x: (x - 1, x, x + 1), points)
    sub_blocks = [mini_block for blocks in block_coordinates for mini_block in blocks]
    dimensions = [{'level': 1, 'blocks': []}]
    for sub_block in sub_blocks:
        for i, dimension in enumerate(dimensions, 1):
            if sub_block not in dimension['blocks']:
                dimension['blocks'].append(sub_block)
                break
            if sub_block in dimension['blocks'] and i == len(dimensions):
                dimensions.append({'level': dimension['level'] + 1, 'blocks': [sub_block]})
                break
    return dimensions


