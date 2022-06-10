
def header(name,parameters:dict):
    r = f'\nSimulation of {name} using QAOA\n\n  PARAMETERS:\n'
    for key in list(parameters.keys()):
        r+= f'\t{key}: {parameters.get(key)}\n'
    print(r)

def footer(starttime,endtime):
    r = f'\n'
    r += calc_time(starttime,endtime)
    print(r)

def calc_time(starttime, endtime):
    deltaT = endtime - starttime
    res = "The program ran for "
    days = deltaT//(3600*24) 
    deltaT = deltaT%(3600*24)
    hours =  deltaT//3600
    deltaT = deltaT%3600
    minutes = deltaT//60
    segs = deltaT%60
    units = 'dhms'
    times = [days,hours,minutes,segs]
    for i,t in enumerate(times):
        if t>0:
            res += f'{t} {units[i]} '
    return res
