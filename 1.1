%%time

import collections
import numpy as np
from numpy import exp
import pandas as pd
import math 
import matplotlib.pyplot as pl
import random

df = pd.read_excel('3.Contact.xlsx')
c = np.array(df)
print("Contact matrix:\n", c, "\n")

df = pd.read_excel('3.1.xlsx')
f = np.array(df).round(2)
print("3.1.xlsx:\n", f, "\n")

df = pd.read_excel('3.2.xlsx')
u = np.array(df).round(2)
print("3.2.xlsx:\n", u, "\n")

df = pd.read_excel('3.2noremote.xlsx')
noru = np.array(df).round(2)
print("no_remote_u.xlsx:\n", noru)

def sum1(input):
    return sum(map(sum, input))
def norm(x):
    # normalise x to range [-1,1]
    nom = (x - x.min())*2
    denom = x.max() - x.min()
    return  (nom/denom) - 1
def sigmoid(x, k):
    return 1/(1+exp(-x/k))
def compress(uncompressed):
    c = np.zeros((run, d_s))
    x = 0
    y = 0
    n = 0
    while x < run:
        while n < (len(john)):
            c[x,y] += uncompressed[x,((d_s*john[n])+y)]
            n+=1
        y += 1
        n = 0
        if y == d_s:
            x += 1
            y = 0
    return c

#USER ENTRY:
component = ["C1","S1","S2","C2","W1","W2","W3","W4","B1","B2","N1","N2","C3","C4","B3","W5","W6","N3","B4","B5","B6","B7"]
direction = ["x+", "x-", "y+", "y-", "z+", "z-"]
prob_array_min = [0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.00]
prob_array_max = [0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25]

d_s = 6 #directions
R = 5 #Run_size
Pc = 5 #Penalty cost
sigk = 0.27
#USER ENTRY END.

(run, rise) = f.shape #rows, columns
frm1 = np.zeros((run,rise)) #copy1 for ranking sequences
frm2 = np.zeros((run,rise)) #copy2 for ranking sequences
frm1 += f

john = list(range(len(component)))
print(john)

fi = compress(f)
u2 = compress(u)
no_ru = compress(noru)

print("\ninital Feasibility: sum:", sum1(fi).round(3), "\n", pd.DataFrame(data=fi, index=component, columns=direction))
print("\ninitial Confidence:\n", pd.DataFrame(data=u2, index=component, columns=direction), "\n")
print("\ninitial Confidence no remote:\n", pd.DataFrame(data=no_ru, index=component, columns=direction), "\n")
pd.DataFrame(data=fi.round(3), index=component, columns=direction).to_csv("ini_f2.csv")
pd.DataFrame(data=u2.round(3), index=component, columns=direction).to_csv("ini_u2.csv")
norm_fi = norm(fi)
norm_u2 = norm(u2)
pd.DataFrame(data=sigmoid(norm_fi,sigk).round(2), index=component, columns=direction).to_csv("ini_sig_f2.csv")
pd.DataFrame(data=sigmoid(norm_u2,sigk).round(2), index=component, columns=direction).to_csv("ini_sig_u2.csv")

run_size = 0 #initial
total_agg_u = np.zeros((run,rise)) #sum of all u matrices, after each cycle
total_agg_f = np.zeros((run,rise)) #sum of all f matrices, after each cycle
total_agg_noru = np.zeros((run,rise)) #sum of all f matrices, after each cycle
mean_agg_u = np.zeros((run,rise)) #^^divided by number of cycles
mean_agg_noru = np.zeros((run,rise)) #^^divided by number of cycles
mean_agg_f = np.zeros((run,rise)) #^^divided by number of cycles
aggregate_u = np.zeros((run,rise))
agg_penalty = np.zeros((run,rise))
f_store = np.zeros((run,rise))
c_store = np.zeros((run,rise))
no_ru_store = np.zeros((run,rise))
no_ru_store += noru
f_store += f
c_store += c

entire_passes = 0
entire_fails = 0
r_comps = []
f_comps = []

r_comps_list = []
r_comps_direction_list = []
r_comps_direction_sequences_list = []
#f_comps_events = []
f_comps_list = []

total_f_score = []
total_u_score = []
total_fxu_score = []

while True:
    if run_size > 0: #misses first pass
        entire_passes += passes
        entire_fails += total_fails
        
        r_comps_list += [r_comps] #creating total sequences list of 'lists'
        r_comps_direction_list += r_comps_direction
        r_comps_direction_sequences_list += [r_comps_direction]
        
        #f_comps_events.append([run_size, f_comps])
        f_comps_list += f_comps #List of all faluires
        
        total_f_score += [f_score] #ranking system list
        total_u_score += [u_score]
        total_fxu_score += [fxu_score]
    
        total_agg_f += aggregate_f
        
        total_agg_noru += agg_noru
        mean_agg_noru = total_agg_noru / run_size
        dup_mean_agg_u = np.zeros((run, rise))
        dup_mean_agg_u += mean_agg_noru #duplicate of mean agg_u
        non_zeros = dup_mean_agg_u != 0 
        dup_mean_agg_u[non_zeros] = 1 #dup_mean_u is now identity matrix of confidence
        ones_count = np.count_nonzero(non_zeros == 1) #Counting one's in dup_mean_u (identity matrix) to divide agg penalty by for steady state balancer
        f += (f_store + agg_penalty) - ((sum1(agg_penalty)/ones_count)*dup_mean_agg_u)#taking away equivalent value of penalties, to make room for fresh variable penalties
        
        f_store = np.zeros((run,rise)) #Re-setting F_store
        agg_penalty = np.zeros((run,rise)) #Re-setting agg_penalty
        f_store += f #Storing new F 
        noru += no_ru_store #re-storing no remote u matrix
        total_agg_u += aggregate_u 
        mean_agg_u = total_agg_u / run_size #gathering mean u so far, to then implement as current u
        
    run_size += 1    
    u += mean_agg_u #initially no affect, later mean-a-u becomes u
    c = np.zeros((run,rise))
    c += c_store #re-storing contact relationships
    frm2 += frm1 #ranking uses a copy of initial feasiiblity: which will be removed piece by piece for online ranking
    
    if run_size == R:
        
        print("Entire Passes/Fails: ", entire_passes, "/", entire_fails)
        #print("\nAll sequences:\n", r_comps_list)
        mean_agg_f = total_agg_f / run_size
        mean_agg_noru = total_agg_noru / run_size
        
        comp_mean_agg_f = compress(mean_agg_f)
        comp_mean_agg_u = compress(mean_agg_u)
        comp_mean_agg_noru = compress(mean_agg_noru)
        
        #print("\ntotal_agg_f after", run_size, "cycles:\n", total_agg_f.round(3))
        #print("\ntotal_agg_u after", run_size, "cycles:\n", total_agg_u.round(3))
        print("\ncomp_mean_agg_f sum:", sum1(comp_mean_agg_f).round(3), "after", run_size, "cycles:\n", pd.DataFrame(data=comp_mean_agg_f.round(3), index=component, columns=direction))
        print( "\n\ncomp_mean_agg_u after", run_size, "cycles:\n", pd.DataFrame(data=comp_mean_agg_u.round(3), index=component, columns=direction))
        print( "\n\ncomp_mean_agg_no_remote_u after", run_size, "cycles:\n", pd.DataFrame(data=comp_mean_agg_noru.round(3), index=component, columns=direction))
        
        norm_mean_f = norm(comp_mean_agg_f)
        norm_mean_u = norm(comp_mean_agg_u)
        print("\nsigmoid_of_mean_agg_f after", run_size, "cycles: (0 being the most removable, 1 being interlocked):\n", pd.DataFrame(data=sigmoid(norm_mean_f,sigk).round(2), index=component, columns=direction))
        print("\nsigmoid_of_mean_agg_u after", run_size, "cycles: (0 being the confident, 1 being uncertain):\n", pd.DataFrame(data=sigmoid(norm_mean_u,sigk).round(2), index=component, columns=direction))
        pd.DataFrame(data=sigmoid(norm_mean_f,sigk).round(2), index=component, columns=direction).to_csv("sig_f2b.csv")
        pd.DataFrame(data=sigmoid(norm_mean_u,sigk).round(2), index=component, columns=direction).to_csv("sig_u2b.csv")
        pd.DataFrame(data=comp_mean_agg_f.round(3), index=component, columns=direction).to_csv("comp_mean_f2b.csv")
        pd.DataFrame(data=comp_mean_agg_u.round(3), index=component, columns=direction).to_csv("comp_mean_u2b.csv")
        pd.DataFrame(data=comp_mean_agg_noru.round(3), index=component, columns=direction).to_csv("comp_mean_no_ru2b.csv")
        
        top_3 = sorted(total_f_score)[:3] #calculating the top 3 f scores and indexes
        index = []
        for k in top_3:
            index += [(total_f_score.index(k))]
        print("\ntop_3 and indicies:", top_3, index, "\ntop 3 sequences based on 'f_score' i.e. the sum of the feasibility of all suggested parts which pass:\n", r_comps_list[index[0]], "\n", r_comps_list[index[1]], "\n", r_comps_list[index[2]])
        
        top_3u = sorted(total_u_score)[:3]
        undex = []
        for l in top_3u:
            undex += [(total_u_score.index(l))]
        print("\ntop_3u and indexes:", top_3u, undex, "\ntop_3u sequences based on 'u_score' i.e. the sum of the uncertaity of all parts which pass:\n", r_comps_list[undex[0]], "\n", r_comps_list[undex[1]], "\n", r_comps_list[undex[2]])
        
        top_3fu = sorted(total_fxu_score)[:3]
        fundex = []
        for w in top_3fu:
            fundex += [(total_fxu_score.index(w))]
        print("\ntop3_fu and indicies:", top_3fu, fundex, "\ntop_3f*u sequences based on 'f*u score':\n", r_comps_list[fundex[0]], "\n", r_comps_list[fundex[1]], "\n", r_comps_list[fundex[2]])
        
        r_comps_dict = collections.Counter([tuple(x) for x in r_comps_list])
        total_f_comps_dict = collections.Counter([f for f in f_comps_list if f])
        r_comps_direction_dict = collections.Counter([f for f in r_comps_direction_list if f])
        r_comps_direction_sequences_dict = collections.Counter([tuple(x) for x in r_comps_direction_sequences_list])
        print("\nDictionary of failed_comps:\n", total_f_comps_dict)
        print("\nDictionary of r_comps sequences:\n", r_comps_dict)
        #print("\nFailed comps events:\n", f_comps_events)
        print("\nDictionary of r_comps_directions:\n", r_comps_direction_dict)
        print("\nDictionary of r_comps sequences with directions:\n", r_comps_direction_sequences_dict)
        print("END")
        break
        
    passes = 0
    fails = 0
    total_fails = 0
    r_comps_direction = []
    r_comps = []
    f_comps = []
    #o_comps = []
    total = [[1,0,0],[0,0,0]]
    coords = 0
    count_components = 0
    aggregate_u = np.zeros((run,rise))
    aggregate_f = np.zeros((run,rise))
    agg_noru = np.zeros((run,rise))
    recursion_stopper = np.zeros((run, d_s))
    f_score = 0
    u_score = 0
    fxu_score = 0

    while True:
        #if run_size > 0:
            #print("\n Sum of summation:", sum1(total))
        if sum1(total) == 0: 
            for item in component:
                if not item in r_comps:
                    r_comps.append(item)
                    #o_comps.append(item + ":P")
            #print("passes, total fails", passes, total_fails)
            #print(r_comps, "removed in chronological order")
            #print(f_comps, "failed")
            #print(o_comps, "Disassembly story")
            #print("\n\n\n*********************   END of CYCLE:  *************************", run_size, "\n\n\n")
            break

        while True:
            r = np.zeros((run, run))
            x = 0
            y = 0
            while x < run:
                r[x,y] = c[x,(d_s*y)] or c[x,(d_s*y)+1] or c[x,(d_s*y)+2] or c[x,(d_s*y)+3] or c[x,(d_s*y)+4] or c[x,(d_s*y)+5]
                y += 1
                if y == run:
                    x += 1
                    y = 0
            #print("\nrelation matrix:\n", r) #taking rowXcolumn contact matrix and scaling to rowXrow relation matrix with OR relation
            
            s1 = np.zeros((run, rise))
            s2 = np.zeros((run, rise))
            N = 1000
            i = 0 
            j = 0
            while i < run:
                dist = u[i,j]*np.random.normal(size=N) #iterating 10000x within confidence tolerance - Monte-carlo propagation
                s1[i,j] = dist.mean() #S1 - mean of normal distrobution for U
                s2[i,j] = 2*dist.std() #S2 - standard deviation
                j += 1
                if j == rise:
                    i += 1
                    j = 0
            #print("\n S1. mean of 10000 samples:\n", s1.round(3), "\n\n S2. Uncertainty± 2*standard deviations:\n", s2.round(3))
            
            fi = compress(frm2) #Compressed initial F
            f1 = compress(f) #Compressed SS Feasability
            s1_1 = compress(s1) #Compressed Mean of 10000 samples(row x directions)
            s2_2 = compress(s2) #Compressed Mean Agg Uncertainty± 2*standard deviation
            
            total = f1 + s1_1 + s2_2 + recursion_stopper #summing the feasibility, uncertainty mean & standard deviation arrays + recusion stopper 
            recursion_stopper = np.zeros((run, d_s))
            np.set_printoptions(suppress=True)
            #print("\npasses;fails;totalfails:", passes, fails, total_fails, "\nSummation matrix: (f1 + s1_1 + s2_2 + recursion_stopper)\n", total.round(2)) 
            if sum1(total) == 0: #when there is one component left sum of summation will equal zero
                break
            #print(*np.where(total == np.min(total[np.nonzero(total)])), np.min(total[np.nonzero(total)]))# minval = np.min(total[np.nonzero(total)])
            #x value here is vital to the "pass" stage

            [x,y] = np.where(total == np.min(total[np.nonzero(total)]))
            index = np.random.randint(0,len(x))
            x = x[index]
            y = y[index]
            #print("\n\nparts already removed:", r_comps, "\nparts failed ", f_comps, "\nOverall Story:", o_comps)
            #print("\n---------Removing component:", component[x], "in direction", direction[int(y)])

            a = int(x*d_s)
            [j] = np.where(r[x] == 1) #j is a list of row coords in relation where 1's exist
            thex = x #copying x & y for later use
            they = y 

            count_components = 0 #counts contacting components
            coords = [] #coordinates of contacting relations
            for z in j:
                count_components += 1
                coords.append([int(x),z])
                coords.append([z,int(x)]) #need both [x,y] and [y,x]
            #print([int(x),z], [z,int(x)], count_components)

            ys = [d_s*g[1] for g in coords] #Compiling and scaling y values from size row to rise
            xs = [x[0] for x in coords] #list comprehension, g is used to represent y here
            all_ys = []
            for n in ys:
                all_ys += n, n+1, n+2, n+3, n+4, n+5

            i = 0 #used to cycle through the associated y values
            confidence_coords = [] #coords used to update the confidence matrix
            for x in xs:
                confidence_coords += [x,all_ys[i]], [x ,all_ys[i+1]], [x ,all_ys[i+2]], [x, all_ys[i+3]], [x ,all_ys[i+4]], [x, all_ys[i+5]]
                i += d_s #x+, x-, y+, y-, z+, z-

            #print(thex, prob_array_min[0], prob_array_max[0]) #looking at the 'training data' to see pass or fail
            prob = random.uniform(prob_array_min[thex], prob_array_max[thex])
            rand_dec = random.random() #random number between 0 and 1
            decision = rand_dec > prob

            if decision is False: #Fail
                #print(decision, "min/max:", prob_array_min[thex], prob_array_max[thex],"random_dec:", rand_dec, "chosen_faluire_prob:", prob)
                #print("thex, they, x,y:",thex, they,  x, y)
                f_comps.append(component[int(thex)]) #appending failed comp to failed comps list
                #o_comps.append(component[int(thex)] + ":F") #appending failed comp to the story
                #print("parts failed:", f_comps)
                fails += 1
                total_fails += 1

                penalty_matrix = np.zeros((run,rise))
                penalty_cost = Pc #Penalty_cost: i.e. the Weight applied
                for [a,b] in coords:
                    penalty_matrix[a,(b*d_s):(b*d_s+d_s)] += u[a,(b*d_s):(b*d_s+d_s)] #copying over all contacting parts confidence into blank matrix 
                penalty_matrix = penalty_matrix*penalty_cost
                agg_penalty += penalty_matrix #storing all penalties to add to initial f in next cycle 
                
                comp_p_matrix = compress(penalty_matrix)
                #print("\n", "compressed penalty matrix:\n", comp_p_matrix)
                f += penalty_matrix

                NIS = total + comp_p_matrix  #Next iteration summation
                NIS_sorted = sorted(NIS[np.nonzero(NIS)].ravel())
                if NIS_sorted[0] in {NIS[thex, 0], NIS[thex, 1], NIS[thex, 2], NIS[thex, 3], NIS[thex, 4], NIS[thex, 5]}:
                    recursion_stopper[thex] += 500 #to the whole component row
                
                for z,p in confidence_coords:
                    if u[z,p] != 0:
                        u[z,p] = u[z,p]*exp(0.01/u[z,p]) #the function deciding the increase in uncertainty, takes into account of initail value of u[x,y]
                    if noru[z,p] != 0:
                        noru[z,p] = noru[z,p]*exp(0.01/noru[z,p]) #doing the same for no remote u for clarity in end results
                #print("confidence_coords::", confidence_coords)
                #print("\nNew uncertainty matrix updated for faluire of", component[int(thex)], "\ncurrent fail-iterations:", fails, ", total fails:", total_fails)
                #print("Overall story:", o_comps, "\n", u.round(3))

            if decision is True: #Pass
                f_score += fi[thex, they] #initial F matrix with parts removed and compressed, not a dynamic matrix, will allow for unique scores and creative solutions
                #print("F_SCORE:", f_score)
                u_score += s2_2[thex, they] #scoring 2*standard deviation of mean confidence
                #print("U_SCORE:", u_score)
                fxu_score += (fi[thex, they])*(s2_2[thex, they])
                #print(decision, "min/max:", prob_array_min[thex], prob_array_max[thex],"random_dec:", rand_dec, "chosen_faluire_prob:", prob)
                r_comps.append(component[int(thex)])
                #o_comps.append(component[int(thex)] + ":P")
                #print("parts removed:", r_comps)
                r_comps_direction.append(component[int(thex)] + direction[int(they)])
                passes += 1
                fails = 0

                for x,g in confidence_coords: #taking a blanket 10% off u[x,g] for passing
                    if u[x,g] >= 1/90:
                        u[x,g] = u[x,g]*0.9
                    if noru[x,g] >= 1/90:
                        noru[x,g] = noru[x,g]*0.9

                aggregate_u += u #adding the entire u matrix to the aggregate matrix
                aggregate_f += f #adding the entire f matrix to the aggregate matrix
                agg_noru += noru #'' No Remote U 
                
                frm2[thex] = 0 #used as ranking feasibilty matrix
                for col in frm2:
                    col[a:(a+d_s)] = 0
                    
                noru[thex] = 0
                for col in noru:
                    col[a:(a+d_s)] = 0
                #print("\nSequence", passes, "NO REMOTE uncertainty matrix after component", component[int(thex)], "removed:\n", u)
            
                f[thex] = 0
                for col in f:
                    col[a:(a+d_s)] = 0
                #print("\nSequence", passes, "Feasability matrix after component", component[int(thex)], "removed:\n", f)

                u[thex] = 0
                for col in u:
                    col[a:(a+d_s)] = 0
                #print("\nSequence", passes, "Uncertainty matrix after component", component[int(thex)], "removed:\n", u)

                c[thex] = 0
                for col in c:
                    col[a:(a+d_s)] = 0
                #print("\nSequence", passes, "contact matrix after", component[int(thex)], "removed:\n", c)

                aggregate_u -= u #removing 'confidence' of components that are yet to be removed
                aggregate_f -= f #removing final 'feasability' of components that are yet to be removed
                agg_noru -= noru
                
                #f_comps_dict = collections.Counter(f for f in f_comps if f)
                #print("\nDictionary of faluires:", f_comps_dict) 
                break
