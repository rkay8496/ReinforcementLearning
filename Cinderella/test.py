def find_adjacent_indices(n, indices):
    # 주어진 인덱스들을 오름차순으로 정렬합니다.
    sorted_indices = sorted(indices)

    # 인접한 인덱스를 저장할 리스트를 초기화합니다.
    adjacent_indices = []

    # 모든 인덱스에 대해 반복합니다.
    for i in range(len(sorted_indices)):
        # 현재 인덱스와 다음 인덱스를 찾습니다.
        current_index = sorted_indices[i]
        next_index = sorted_indices[(i + 1) % len(sorted_indices)]

        # 만약 현재 인덱스와 다음 인덱스가 인접하다면, 이 두 인덱스를 adjacent_indices에 추가합니다.
        if (next_index - current_index) % n == 1 or (current_index - next_index) % n == 1:
            if current_index not in adjacent_indices:
                adjacent_indices.append(current_index)
            if next_index not in adjacent_indices:
                adjacent_indices.append(next_index)

    # 인접한 인덱스의 리스트를 반환합니다.
    return adjacent_indices

print(find_adjacent_indices(5, [0, 2, 4]))