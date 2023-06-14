

def linear(interval=1, offset=0):
    # print(interval)
    # print(offset)

    docstring = "Track at iterations {" + f"{offset} + n * {interval} " + "| n >= 0}."

    def schedule(global_step):
        shifted = global_step - offset
        if shifted < 0:
            return False
        else:
            return shifted % interval == 0

    schedule.__doc__ = docstring

    return schedule


schedulers = {
    'linear': linear
}


class ScheduleSelector:
    """
    静态类， 可以通过str获取Schedule
    好处：
    """
    @staticmethod
    def select(schedule_name):
        if len(schedule_name) == 0:
            return linear()
        schedule, augs = ScheduleSelector.parse_schedule(schedule_name)
        if schedule not in schedulers.keys():
            raise NotImplementedError(
                "hook not found: {}".format(schedule))
        schedule = schedulers[schedule](*augs)
        return schedule
    
    @staticmethod
    def parse_schedule(schedules):
        schedule = schedules[:schedules.index("(")].strip()
        schedule_augument = schedules[schedules.index("(")+1: -1].strip()
        augs = [int(x.strip()) for x in schedule_augument.split(',')]
        return schedule, augs
        

if __name__ == '__main__':
    name = 'linear(2, 0)'
    ScheduleSelector.select(name)
