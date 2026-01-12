import random
import numpy as np


def valid_crop_resize(data_numpy, valid_frame_num, p_interval, window):
    """
    Crop and resize the valid frames from the input data.

    Args:
        data_numpy: Input data with shape (C, T, V, M)
        valid_frame_num: List of valid frame numbers for each sample
        p_interval: Probability interval for sampling
        window: Target window size

    Returns:
        Cropped and resized data
    """
    # input: C,T,V,M00 211
    C, T, V, M = data_numpy.shape
    begin = 0
    end = valid_frame_num
    valid_size = end - begin

    # crop
    if len(p_interval) == 1:
        p = p_interval[0]
        bias = int((1 - p) * valid_size / 2)
        data = data_numpy[:, begin + bias:end - bias, :, :]  # center crop
        cropped_length = data.shape[1]
    else:
        p = np.random.rand(1) * (p_interval[1] - p_interval[0]) + p_interval[0]
        cropped_length = np.minimum(np.maximum(int(np.floor(valid_size * p)), 64),
                                    valid_size)  # constraint cropped_length lower bound as 64
        bias = np.random.randint(0, valid_size - cropped_length + 1)
        data = data_numpy[:, begin + bias:begin + bias + cropped_length, :, :]
        if data.shape[1] == 0:
            print(cropped_length, bias, valid_size)

    # resize
    data = resize(data, cropped_length, window)

    return data


def resize(data, cropped_length, window):
    """
    Resize the cropped data to the target window size.

    Args:
        data: Input data with shape (C, T, V, M)
        cropped_length: Length after cropping
        window: Target window size

    Returns:
        Resized data
    """
    # input: C,T,V,M
    C, T, V, M = data.shape
    data_resized = np.zeros((C, window, V, M))

    if cropped_length < window:
        # Interpolate when cropped length is smaller than window
        indices = np.linspace(0, cropped_length - 1, window).astype(int)
        data_resized = data[:, indices, :, :]
    else:
        # Sample when cropped length is larger than or equal to window
        indices = np.linspace(0, cropped_length, window + 1).astype(int)[:-1]
        data_resized = data[:, indices, :, :]

    return data_resized


def random_choose(data_numpy, size, auto_pad=True):
    """
    Randomly choose a clip from the input data.

    Args:
        data_numpy: Input data with shape (C, T, V, M)
        size: Target size
        auto_pad: Whether to automatically pad if data is shorter than size

    Returns:
        Randomly chosen clip
    """
    # input: C,T,V,M
    C, T, V, M = data_numpy.shape
    if T == size:
        return data_numpy
    elif T < size:
        if auto_pad:
            return auto_pading(data_numpy, size)
        else:
            return data_numpy
    else:
        begin = random.randint(0, T - size)
        return data_numpy[:, begin:begin + size, :, :]


def random_move(data_numpy,
                angle_candidate=[-10., -5., 0., 5., 10.],
                scale_candidate=[0.9, 1.0, 1.1],
                transform_candidate=[-0.2, -0.1, 0.0, 0.1, 0.2],
                move_time_candidate=[1]):
    """
    Randomly rotate, scale and translate the input data.

    Args:
        data_numpy: Input data with shape (C, T, V, M)
        angle_candidate: List of rotation angles (degrees)
        scale_candidate: List of scaling factors
        transform_candidate: List of translation values
        move_time_candidate: List of number of transformations to apply

    Returns:
        Transformed data
    """
    # input: C,T,V,M
    C, T, V, M = data_numpy.shape
    move_time = random.choice(move_time_candidate)
    node = np.arange(0, T, T * 1.0 / move_time).round().astype(int)
    node = np.append(node, T)
    num_node = len(node)

    A = np.random.choice(angle_candidate, num_node)
    S = np.random.choice(scale_candidate, num_node)
    T_x = np.random.choice(transform_candidate, num_node)
    T_y = np.random.choice(transform_candidate, num_node)

    a = np.zeros(T)
    s = np.zeros(T)
    t_x = np.zeros(T)
    t_y = np.zeros(T)

    # linspace
    for i in range(num_node - 1):
        a[node[i]:node[i + 1]] = np.linspace(
            A[i], A[i + 1], node[i + 1] - node[i]) * np.pi / 180
        s[node[i]:node[i + 1]] = np.linspace(S[i], S[i + 1],
                                             node[i + 1] - node[i])
        t_x[node[i]:node[i + 1]] = np.linspace(T_x[i], T_x[i + 1],
                                               node[i + 1] - node[i])
        t_y[node[i]:node[i + 1]] = np.linspace(T_y[i], T_y[i + 1],
                                               node[i + 1] - node[i])

    theta = np.array([[np.cos(a) * s, -np.sin(a) * s],
                      [np.sin(a) * s, np.cos(a) * s]])  # rotation + scale

    # perform transformation
    for i_frame in range(T):
        xy = data_numpy[0:2, i_frame, :, :]
        new_xy = np.dot(theta[:, :, i_frame], xy.reshape(2, -1))
        new_xy[0] += t_x[i_frame]
        new_xy[1] += t_y[i_frame]
        data_numpy[0:2, i_frame, :, :] = new_xy.reshape(2, V, M)

    return data_numpy


def random_shift(data_numpy):
    """
    Randomly shift the input data in time.

    Args:
        data_numpy: Input data with shape (C, T, V, M)

    Returns:
        Shifted data
    """
    # input: C,T,V,M
    C, T, V, M = data_numpy.shape
    data_shift = np.zeros(data_numpy.shape)
    valid_frame = (data_numpy != 0).sum(axis=3).sum(axis=2).sum(axis=0) > 0
    begin = valid_frame.argmax()
    end = len(valid_frame) - valid_frame[::-1].argmax()

    size = end - begin
    bias = random.randint(0, T - size)
    data_shift[:, bias:bias + size, :, :] = data_numpy[:, begin:end, :, :]

    return data_shift


def openpose_match(data_numpy):
    """
    Match and sort keypoints from OpenPose.

    Args:
        data_numpy: Input data with shape (C, T, V, M)

    Returns:
        Matched and sorted data
    """
    C, T, V, M = data_numpy.shape
    assert (C == 3)
    score = data_numpy[2, :, :, :].sum(axis=1)
    # the rank of body confidence in each frame (shape: T-1, M)
    rank = (-score[0:T - 1]).argsort(axis=1).reshape(T - 1, M)

    # data of frame 1
    xy1 = data_numpy[0:2, 0:T - 1, :, :].reshape(2, T - 1, V, M, 1)
    # data of frame 2
    xy2 = data_numpy[0:2, 1:T, :, :].reshape(2, T - 1, V, 1, M)
    # square of distance between frame 1&2 (shape: T-1, M, M)
    distance = ((xy2 - xy1) ** 2).sum(axis=2).sum(axis=0)

    # match pose
    forward_map = np.zeros((T, M), dtype=int) - 1
    forward_map[0] = range(M)
    for m in range(M):
        choose = (rank == m)
        forward = distance[choose].argmin(axis=1)
        for t in range(T - 1):
            distance[t, :, forward[t]] = np.inf
        forward_map[1:][choose] = forward
    assert (np.all(forward_map >= 0))

    # string data
    for t in range(T - 1, 0, -1):
        forward_map[t] = forward_map[t - 1, forward_map[t]]

    # generate data
    new_data_numpy = np.zeros(data_numpy.shape)
    for t in range(T):
        new_data_numpy[:, t, :, :] = data_numpy[:, t, :, forward_map[t]].transpose(1, 2, 0)
    data_numpy = new_data_numpy

    # score sort
    trace_score = data_numpy[2, :, :, :].sum(axis=1).sum(axis=0)
    rank = (-trace_score).argsort()
    data_numpy = data_numpy[:, :, :, rank]

    return data_numpy


def top_k_by_category(label, score, top_k):
    """
    Select top k predictions by category.

    Args:
        label: Ground truth labels
        score: Prediction scores
        top_k: Number of top predictions to consider

    Returns:
        Accuracy
    """
    instance_num, class_num = score.shape
    rank = score.argsort()
    hit_top_k = [[] for i in range(top_k)]
    for i in range(instance_num):
        l = label[i]
        r = rank[i, -top_k:]
        for j in range(top_k):
            if r[j] == l:
                hit_top_k[j].append(i)

    accuracy_list = []
    for i in range(top_k):
        accuracy_list.append(len(hit_top_k[i]) * 1.0 / instance_num)
    return accuracy_list


def calculate_recall_precision(label, score):
    """
    Calculate recall and precision.

    Args:
        label: Ground truth labels
        score: Prediction scores

    Returns:
        Tuple of (recall, precision)
    """
    instance_num, class_num = score.shape
    rank = score.argsort()
    prediction = rank[:, -1]

    # recall
    recall_list = [0] * class_num
    for i in range(instance_num):
        l = label[i]
        if l == prediction[i]:
            recall_list[l] += 1

    # precision
    precision_list = [0] * class_num
    for i in range(instance_num):
        p = prediction[i]
        if label[i] == p:
            precision_list[p] += 1

    return recall_list, precision_list


def auto_pading(data_numpy, size, random_pad=False):
    """
    自动填充或裁剪数据到指定大小

    Args:
        data_numpy: 输入数据，形状为 (C, T, V, M)
        size: 目标时间帧数
        random_pad: 是否随机填充（暂未实现）

    Returns:
        处理后的数据，形状为 (C, size, V, M)
    """
    C, T, V, M = data_numpy.shape

    if T == size:
        return data_numpy
    elif T < size:
        # 需要填充
        pad_len = size - T
        # 重复最后一帧进行填充
        pad_data = np.repeat(data_numpy[:, [-1], :, :], pad_len, axis=1)
        data_numpy = np.concatenate([data_numpy, pad_data], axis=1)
    else:
        # 需要裁剪
        begin = (T - size) // 2
        data_numpy = data_numpy[:, begin:begin + size, :, :]

    return data_numpy


# 添加正确拼写的别名
auto_padding = auto_pading
def random_rot(data_numpy, theta=0.3):
    """Randomly rotate skeleton around z-axis"""
    import random
    data_numpy = np.array(data_numpy)
    C, T, V, M = data_numpy.shape
    angle = (random.random() - 0.5) * 2 * theta
    cos_a, sin_a = np.cos(angle), np.sin(angle)
    R = np.array([[cos_a, -sin_a, 0], [sin_a, cos_a, 0], [0, 0, 1]])
    for t in range(T):
        for m in range(M):
            data_numpy[:, t, :, m] = np.dot(R, data_numpy[:, t, :, m])
    return data_numpy


# ============ AimCLR Augmentation Functions ============

def random_rot(data_numpy, theta=0.3):
    """Randomly rotate skeleton around z-axis"""
    import random
    data_numpy = np.array(data_numpy)
    C, T, V, M = data_numpy.shape
    angle = (random.random() - 0.5) * 2 * theta
    cos_a, sin_a = np.cos(angle), np.sin(angle)
    R = np.array([[cos_a, -sin_a, 0], [sin_a, cos_a, 0], [0, 0, 1]])
    for t in range(T):
        for m in range(M):
            data_numpy[:, t, :, m] = np.dot(R, data_numpy[:, t, :, m])
    return data_numpy

def shear(data_numpy, s1=0.5, s2=0.5):
    """Apply shear transformation"""
    import random
    data_numpy = np.array(data_numpy)
    C, T, V, M = data_numpy.shape
    s1_val = random.uniform(-s1, s1)
    s2_val = random.uniform(-s2, s2)
    S = np.array([[1, s1_val, 0], [s2_val, 1, 0], [0, 0, 1]])
    for t in range(T):
        for m in range(M):
            data_numpy[:, t, :, m] = np.dot(S, data_numpy[:, t, :, m])
    return data_numpy

def temperal_crop(data_numpy, temperal_padding_ratio=6):
    """Temporal cropping with padding"""
    import random
    C, T, V, M = data_numpy.shape
    if temperal_padding_ratio <= 0:
        return data_numpy
    min_len = max(int(T * 0.5), 1)
    crop_len = random.randint(min_len, T)
    if crop_len < T:
        start = random.randint(0, T - crop_len)
        cropped = data_numpy[:, start:start+crop_len, :, :]
        pad_l = (T - crop_len) // 2
        pad_r = T - crop_len - pad_l
        cropped = np.pad(cropped, ((0,0), (pad_l,pad_r), (0,0), (0,0)), mode='edge')
        return cropped
    return data_numpy

