
from CTILP_Optimization_Baltimore_TestComplexity_ver02_profile import *

@profile
def set_up_data(radius_, file_name):

    # model
    m = OSMNX_Map_ILP(file_name, radius=radius_, same=True, Baltimore =False)
    return m

@profile
def set_up_model(m ):

    m.initial_ILP()
    m.update_model_OSMNX(distance_OSMNX,affect_OSMNX,
                                 power = 1, d_e = 240, model = 2)

    return m


if __name__ == '__main__':

    # map radius (meters)
    # Total Budget (USD)
    radius = 1300
    Budget = 500000
    file_name = '_1516KenhillAve1300_123017'
    model_type = 'bigM'
    power = 'power1'
    effective_distance = 240
    normal = 'nonnormal'

    m = set_up_data(radius, file_name)
    m = set_up_model(m)

    # Optimization
    for i in xrange(1):
        m.solve()
        m.plot(m.x, size = 10, name = model_type + "-" + power +"-distance" + str(effective_distance) + "-" + normal + file_name + "-" + str(i) )
        m.no_good_update()

    print m.status

