"""Deterministic collision-group attempt order with a fixed total budget."""


class RepairSchedule:
    def __init__(self, groups, limit, *, total_limit=None):
        self.order = list(groups)
        self.limit = int(limit)
        self.ready = list(groups)
        self.total_limit = self.limit * len(self.order) if total_limit is None else int(total_limit)
        self.total = 0
        self.attempts = dict.fromkeys(groups, 0)

    def next(self):
        if not self.ready or self.total >= self.total_limit:
            return None
        group = self.ready.pop(0)
        self.attempts[group] += 1
        self.total += 1
        return group, self.attempts[group] - 1

    def peek_first(self, count):
        available = max(0, self.total_limit - self.total)
        return [group for group in self.ready[:available] if self.attempts[group] == 0][:count]

    def retry(self, group):
        if self.attempts[group] < self.limit and group not in self.ready:
            self.ready.append(group)
            self.ready.sort(key=self.order.index)
