import numpy as np


class KMMatcher:

    # weights : nxm weight matrix (numpy , float), n <= m
    def __init__(self, weights):
        weights = np.array(weights).astype(np.float32)
        self.weights = weights
        self.n, self.m = weights.shape
        assert self.n <= self.m
        # init label
        self.label_x = np.max(weights, axis=1)
        self.label_y = np.zeros((self.m,), dtype=np.float32)

        self.max_match = 0
        self.xy = -np.ones((self.n,), dtype=np.int)
        self.yx = -np.ones((self.m,), dtype=np.int)

    def do_augment(self, x, y):
        self.max_match += 1
        while x != -2:
            self.yx[y] = x
            ty = self.xy[x]
            self.xy[x] = y
            x, y = self.prev[x], ty

    def find_augment_path(self):
        self.S = np.zeros((self.n,), np.bool)
        self.T = np.zeros((self.m,), np.bool)

        self.slack = np.zeros((self.m,), dtype=np.float32)
        self.slackyx = -np.ones((self.m,), dtype=np.int)  # l[slackyx[y]] + l[y] - w[slackx[y], y] == slack[y]

        self.prev = -np.ones((self.n,), np.int)

        queue, st = [], 0
        root = -1

        for x in range(self.n):
            if self.xy[x] == -1:
                queue.append(x);
                root = x
                self.prev[x] = -2
                self.S[x] = True
                break

        self.slack = self.label_y + self.label_x[root] - self.weights[root]
        self.slackyx[:] = root

        while True:
            while st < len(queue):
                x = queue[st]
                st += 1

                is_in_graph = np.isclose(self.weights[x], self.label_x[x] + self.label_y)
                nonzero_inds = np.nonzero(np.logical_and(is_in_graph, np.logical_not(self.T)))[0]

                for y in nonzero_inds:
                    if self.yx[y] == -1:
                        return x, y
                    self.T[y] = True
                    queue.append(self.yx[y])
                    self.add_to_tree(self.yx[y], x)

            self.update_labels()
            queue, st = [], 0
            is_in_graph = np.isclose(self.slack, 0)
            nonzero_inds = np.nonzero(np.logical_and(is_in_graph, np.logical_not(self.T)))[0]

            for y in nonzero_inds:
                x = self.slackyx[y]
                if self.yx[y] == -1:
                    return x, y
                self.T[y] = True
                if not self.S[self.yx[y]]:
                    queue.append(x)
                    self.add_to_tree(self.yx[y], x)

    def solve(self, verbose=False):
        while self.max_match < self.n:
            x, y = self.find_augment_path()
            self.do_augment(x, y)

        sum = 0.
        for x in range(self.n):
            if verbose:
                print('match {} to {}, weight {:.4f}'.format(x, self.xy[x], self.weights[x, self.xy[x]]))
            sum += self.weights[x, self.xy[x]]
        self.best = sum
        if verbose:
            print('ans: {:.4f}'.format(sum))
        return self.xy

    def add_to_tree(self, x, prevx):
        self.S[x] = True
        self.prev[x] = prevx

        better_slack_idx = self.label_x[x] + self.label_y - self.weights[x] < self.slack
        self.slack[better_slack_idx] = self.label_x[x] + self.label_y[better_slack_idx] - self.weights[
            x, better_slack_idx]
        self.slackyx[better_slack_idx] = x

    def update_labels(self):
        delta = self.slack[np.logical_not(self.T)].min()
        self.label_x[self.S] -= delta
        self.label_y[self.T] += delta
        self.slack[np.logical_not(self.T)] -= delta


def match(old_tracks, centers, k, verbose=False):
    """
    Used to match the new center with old tracks
    :param verbose: verbose
    :param old_tracks: [[[x11,y11],[x12,y12]],
                        [[x21,y21],[x22,y22]]]
                        every single line represent a track
    :param centers: new detected centers
    :param k: the length of track
    :return: tracks after match with new centers
    """
    if len(centers) == 0:
        return
    if len(old_tracks) == 0:
        for center in centers:
            old_tracks.append([center])
        return old_tracks

    # Get the length of tracks
    def track_length(elem):
        return len(elem)

    # ensure that the numbers of tracks and centers are equal
    number_tracks, number_centers = len(old_tracks), len(centers)
    if number_centers > number_tracks:
        pass
    elif number_centers < number_tracks:
        old_tracks.sort(key=track_length)
        old_tracks = old_tracks[number_tracks - number_centers:]
    if verbose:
        print('-----------------------------------------------------')
        print('old_tracks: ', old_tracks)
        print('centers: ', centers)

    distance_matrix = []
    for track in old_tracks:
        distance = []
        for center in centers:
            if (abs(track[-1][0] - center[0]) + abs(track[-1][1] - center[1])) == 0:
                dis = 10 ** 4
            else:
                dis = int(100 / (abs(track[-1][0] - center[0]) + abs(track[-1][1] - center[1])))
            distance.append(dis)
        distance_matrix.append(distance)
    if verbose:
        print('distance_matrix:', distance_matrix)
    matcher = KMMatcher(distance_matrix)
    solve = matcher.solve(verbose)

    centers_index = list(range(len(centers)))
    for i in range(len(solve)):
        old_tracks[i].append(centers[solve[i]])
        centers_index.remove(solve[i])
        if len(old_tracks[i]) > k:
            # if the track is too long
            old_tracks[i].pop(0)
    # if there till have centers
    if len(centers_index) > 0:
        # which means we find new track
        for index in centers_index:
            old_tracks.append([centers[index]])

    return old_tracks
