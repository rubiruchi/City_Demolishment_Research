from report import *
import os

# load model files path
path='model/'
files = [f[:-4] for f in os.listdir(path) if f.endswith('.mps')]
files.sort()

# load report table
#model_house_df = pd.read_csv('modelsTableexp',index_col=0)
model_house_df = pd.DataFrame(
                              columns=['model','iter','Cost','Num_of_houses','ObjVal','Time','budg'] +
                              list(gdf.ID)+ ["power","distance"])


def run(dis,budg,idx, impact = True, top = 1, p =0,dif = False):
    idx = idx
    for d,b in zip(sorted(dis*len(budg)),budg*len(dis)):
    #for fileN in files:
        fileN = 'modelexp-d{}-p{}'.format(d,p)
        rmodel, rx, ry, rbigM= read_model("model/" + fileN)
        rmodel.params.MIPGap = 0.000000001
        solver = LPSolve(rmodel,rx,ry,rbigM)
        solver.budget_constraint(Budget=b)
        solver.set_objective(impact)
        for i in xrange(1,1+top):
            solver.solve(dif)
            if impact:
                model_house_df.loc[idx] = np.zeros(len(gdf)+9)
                model_house_df.loc[idx,'budg'] = b
                model_house_df.loc[idx,'model'] = fileN
                model_house_df.loc[idx,'iter'] = i
                model_house_df.loc[idx,['Cost','Num_of_houses','ObjVal','Time']] = solver.status[-1]
                model_house_df.loc[idx,[k for k,v in solver.x.iteritems() if abs(v.x -1) < 0.00001 ]] += 1
                model_house_df.loc[idx,'power'] = p
                model_house_df.loc[idx,'distance'] = d
            else:
                model_house_df.loc[idx] = np.zeros(len(gdf)+9)
                model_house_df.loc[idx,'budg'] = b
                model_house_df.loc[idx,'model'] = "base"
                model_house_df.loc[idx,'iter'] = i
                model_house_df.loc[idx,['Cost','Num_of_houses','ObjVal','Time']] = solver.status[-1]
                model_house_df.loc[idx,[k for k,v in solver.x.iteritems() if abs(v.x - 1) < 0.00001 ]] = 1
                model_house_df.loc[idx,'power'] = -1
                model_house_df.loc[idx,'distance'] = -1
            idx+=1
            yield

        #solver.model.write('model/model-d%s-p%s.mps'%(d,p))
        #solver.model.write('model/model-d%s-p%s.prm'%(d,p))


if  __name__ == '__main__':

    #### Impact
    # parameter
    #dis = [400]
    #budg = range(1000000,5000000,1000000) + \
    #        range(6000000,10000000,1000000) + \
    #        range(11000000,20000000,1000000) + \
    #        range(21000000,40000000,1000000)
    #top = 1

    # init generator
    #run_result = run([400],budg,len(model_house_df))
    # run
    #for i in xrange(len(dis)*len(budg)*top):
    #    run_result.next()


    #model_house_df.to_csv('modelsTable')

    #### Base
    # parameter
    for dist in [50,75,100,125,150]:
        dis = [dist]
        budg = [5000000]
        top = 9
        # init generator
        run_result = run(dis,budg,len(model_house_df),True,top,1,True)
        # run
        for i in xrange(len(dis)*len(budg)*top):
            run_result.next()

        model_house_df.to_csv('modelsTableexpTop9difference-d{}-p1'.format(dis))
        model_house_df = model_house_df.iloc[0:0]

    for dist in [50,75,100,125,150]:
        dis = [dist]
        budg = [5000000]
        top = 9
        # init generator
        run_result = run(dis,budg,len(model_house_df),True,top,1)
        # run
        for i in xrange(len(dis)*len(budg)*top):
            run_result.next()

        model_house_df.to_csv('modelsTableexpTop9difference-d{}-p1-n'.format(dis))
        model_house_df = model_house_df.iloc[0:0]
    #run_result = run(dis,budg,len(model_house_df),True,top,2)
    # run
    #for i in xrange(len(dis)*len(budg)*top):
    #     run_result.next()

    #model_house_df.to_csv('modelsTableexpTop9difference')






