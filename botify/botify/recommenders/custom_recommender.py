import numpy as np
from collections import defaultdict

from .random import Random
from .recommender import Recommender


class CustomRecommender(Recommender):

    def __init__(self, tracks_redis, catalog, history_redis, another_recommender=None, max_count_listen=3):
        self.tracks_redis = tracks_redis
        self.fallback = Random(tracks_redis)
        self.another_recommender = another_recommender
        self.catalog = catalog
        self.history_redis = history_redis
        self.max_count_listen = max_count_listen

    def recommend_next(self, user: int, prev_track: int, prev_track_time: float) -> int:
        history_tracks = self.history_redis.get(user)

        if history_tracks is not None:
            history_tracks = self.catalog.from_bytes(history_tracks)
        else:
            # this structure will make it easier to add new info in "history-tracks"
            history_tracks = {
                "count_listened": defaultdict(float)
            }

        previous_track = self.tracks_redis.get(prev_track)
        if previous_track is None:
            return self.fallback.recommend_next(user, prev_track, prev_track_time)

        history_tracks["count_listened"][prev_track] += 0.95 - prev_track_time
        self.history_redis.set(user, self.catalog.to_bytes(history_tracks))

        previous_track = self.catalog.from_bytes(previous_track)
        recommendations = previous_track.recommendations
        weight = previous_track.time_weight
        if not recommendations:
            return self.fallback.recommend_next(user, prev_track, prev_track_time)

        del_ind = []
        for ind, recommend in enumerate(recommendations):
            if history_tracks['count_listened'][recommend] > self.max_count_listen:
                del_ind.append(ind)

        for curr_ind in del_ind:
            recommendations.pop(curr_ind)
            weight.pop(curr_ind)

        if prev_track_time < 0.7 and self.another_recommender is not None or not recommendations:
            return self.another_recommender.recommend_next(user, prev_track, prev_track_time)
        return int(np.random.choice(recommendations, p=np.array(weight) / sum(weight)))
