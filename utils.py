from datetime import datetime

def current_date(now):
    return now.strftime('%Y-%m-%d')

def current_time(now):
    return now.strftime('%Y-%m-%d_%H:%M:%S')