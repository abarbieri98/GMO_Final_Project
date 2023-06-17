import numpy as np
from pedalboard import *
from pedalboard.io import AudioFile
import scipy.signal as sig
import scipy.io.wavfile as open_wav
import matplotlib.pyplot as plt
import os
import librosa
import librosa.display
import IPython.display as ipd
from soundfile import write

path_to_plugin = "C:\Program Files\Common Files\VST3"
flanger = load_plugin(path_to_plugin + "\ValhallaSpaceModulator.vst3")
dly = load_plugin(path_to_plugin + "\ValhallaFreqEcho.vst3")
reverb = load_plugin(path_to_plugin + "\ValhallaSupermassive.vst3")

# Not all the VSTs' parameters are useful for the task to be accomplished. 
# The following lines specify which parameters to take into consideration. 

dly_con = [1,2,3,4,5,6] # Continuous delay parameters
rev_con = [3,4,6,7,8,9,10,11,12] # Continuous reverb parameters

par_con = [list(np.arange(1,5)),dly_con,rev_con] 
mix = [0,0,0] # mix parameters for the three vsts

par_cat = [5,13] # Categorical parameters for flanger and reverb
flanger_modes = ['Up', 'Up/Down', 'TZFlange+', 'TZFlange-', 'TriFlange', 'Ocho', 'Doubler', 'VariUp', 'VariUpDown', 'Ensemble360', 'Symphonic']
reverb_modes = ['Gemini', 'Hydra', 'Centaurus', 'Sagittarius', 'Great Annihilator', 'Andromeda', 'Lyra', 'Capricorn', 'Triangulum', 'Large Magellanic Cloud', 
        'Cirrus Major', 'Cirrus Minor', 'Cassiopeia', 'Orion', 'Aquarius', 'Pisces', 'Gemini'] 
modes = [flanger_modes,reverb_modes]

# Here we define the base pedalboard to use for fitness evaluation of the individuals. 
base_board = Pedalboard([
    flanger,
    dly,
    reverb
])

class spectrogram:

    def from_file(file:str, hop_len = 512):
        """Extract spectrogram from an audio file."""
        test, ts = librosa.load(file,mono=False)
        S_test = librosa.stft(test, n_fft=2048, hop_length=hop_len)
        Y_scale = np.abs(S_test)**2
        Y_log_scale = librosa.power_to_db(Y_scale)
        return Y_log_scale,ts

    def from_array(audio:np.ndarray, hop_len = 512):
        """Extract spectrogram from a NumPy array."""
        S_test = librosa.stft(audio, n_fft=2048, hop_length= hop_len)
        Y_scale = np.abs(S_test)**2
        Y_log_scale = librosa.power_to_db(Y_scale)
        return Y_log_scale

    def plot(Y, sr, hop_length, y_axis="linear", diff=False):
        """Plot spectrogram. Y should be output from stft function."""
        plt.figure(figsize=(25, 10))
        if diff==False:
            librosa.display.specshow(Y, 
                                    sr=sr, 
                                    hop_length=hop_length, 
                                    x_axis="time", 
                                    y_axis=y_axis,
                                    )
        else:
            librosa.display.specshow(Y, 
                                    sr=sr, 
                                    hop_length=hop_length, 
                                    x_axis="time", 
                                    y_axis=y_axis,
                                    cmap="seismic"
                                    )
        plt.colorbar(format="%+2.f")

class utils:
    def minmax_std(val,mi,ma):
        """Applies min-max standardization to get values between 0 and 1."""
        return (val-mi)/(ma-mi)

    def rev_minmax(val,mi,ma):
        """Reverse min-max standardization, transform a scalar in [0,1] back to its original space."""
        return val*(ma-mi)+mi
    def get_parameters(board:Pedalboard)->list:
        """Returns a list of parameters from a pedalboard."""
        parameters = []
        for i in range(3):
            par_tmp = []
            names = list(board[i].parameters.keys())
            for par in par_con[i]:
                value = getattr(board[i],names[par])
                par_tmp.append(value)
            parameters.append(par_tmp)
        return parameters
    def set_parameters(parameters:list,board:Pedalboard):
        """Given a list of admissible parameters, set them to a pedalboard."""
        for i in range(3):
            names = list(board[i].parameters.keys())
            n = len(parameters[0][i])
            # Set continuous parameters
            for j in range(n):
                setattr(board[i],names[par_con[i][j]],parameters[0][i][j])
            # Set mix parameter
            setattr(board[i],names[mix[i]],parameters[2][i])
        # Set categorical parameters
        setattr(board[0],names[par_cat[0]],parameters[1][0])
        setattr(board[2],names[par_cat[1]],parameters[1][1])
        board.reset()
    def clip_signal(audio):
        """Clip the audio signal if nan or inf values are encountered."""
        for i in range(2):
            over_nan = np.where(np.isnan(audio[i]))[0]
            over_inf = np.where(np.isinf(audio[i]))[0]
            for j in over_nan:
                audio[i][j] = 0
            for j in over_inf:
                audio[i][j] = 0       

class individual:
    """Base individual class. An individual is composed by three members:
        * con: a list containing the continuous parameters of the effects.
        * cat: a list containing the mode of the effects (categorical variables)
        * mix: continuous values reflecting the mix between dry sound and effects
        """
    def __init__(self):
        super().__init__()
        self.con = [np.zeros(len(par_con[i])) for i in range(3)]
        self.cat = [0,0,0]
        self.mix = [0,0,0]
        self.parameters = [self.con,self.cat,self.mix]

        bias_effect = np.random.randint(0,3) 
        for i in range(3):
            # mix parameter
            name = list(base_board[i].parameters.keys())[mix[i]]
            parameter = base_board[i].parameters[name]
            top = parameter.range[1]
            bottom = parameter.range[0]
            # Mix is initialized in order to enhance one of the three effects
            if(i == bias_effect):
                new_val = utils.rev_minmax(np.random.uniform(.5,1),bottom,top)
            else:
                new_val = utils.rev_minmax(np.random.uniform(0,.25),bottom,top)
            self.mix[i] = new_val
            lst = par_con[i]
            names = list(base_board[i].parameters.keys())
            # Continuous parameters
            for j in range(len(lst)):
                parameter = base_board[i].parameters[names[lst[j]]]
                top = parameter.range[1]
                bottom = parameter.range[0]
                new_val = utils.rev_minmax(np.random.uniform(),bottom,top)
                self.con[i][j] = new_val
            # Categorical parameters
            if(i<2):
                lst = par_cat[i]
                self.cat[i] = np.random.choice(modes[i],1)[0]

# Note: all the algorithms use a modified version of the Hooke-Jeeves algorithm for local optimization, applied following 
# the mechanism presented in section 3.2 of https://www.sciencedirect.com/science/article/pii/S1568494612001226?via%3Dihub

class Base():
    def __init__(self,input,target,sr, hop_len=512):
        """Initialize DE class.
        
        Parameters
        ---
        *input: array version of input audio
        *target: spectrogram of target audio
        *sr: sampling rate for both input and target audio
        *hop_len: Hop size in STFT"""

        self.input = input
        self.target = target
        self.sr = sr
        self.hop_len = hop_len
    def clip_parameters(self,parameters,board): 
        """Check if parameters are inside the upper and lower bound, substituting invalid ones
            in damping fashion using the formula new_val = invalid_val +/- 2*(invalid_val - max/min_val) """
        for i in range(3):
            lst = par_con[i]
            names = list(board[i].parameters.keys())
            n = len(lst)
            for j in range(n):
                parameter = board[i].parameters[names[lst[j]]]
                diff_top = parameters[i][j] - parameter.range[1]
                diff_bottom = parameter.range[0] - parameters[i][j] 
                if diff_top > 0:
                    parameters[i][j] -= np.random.uniform(1,2)*diff_top 
                if diff_bottom > 0:
                    parameters[i][j] += np.random.uniform(1,2)*diff_bottom

    def unif_crossover(self,original,donor,p=.5):
        """Returns the child between the "original" and "donor" vectors obtained using uniform crossover, 
        ensuring that at least one value is inherited from the donor vector."""
        child = individual()
        # Import from donor
        type_donor = np.random.randint(0,3)
        eff_donor = np.random.randint(0,3)
        par_donor = 0
        if(type_donor == 0):
            par_donor = np.random.randint(0,len(donor.parameters[type_donor][eff_donor]))
            child.parameters[type_donor][eff_donor][par_donor] = donor.parameters[type_donor][eff_donor][par_donor]

        else:
            child.parameters[type_donor][eff_donor] = donor.parameters[type_donor][eff_donor]
        
        for eff in range(3):
            # Continuous
            for j in range(len(child.con[eff])):
                if (eff != eff_donor) and (j != par_donor):
                    r = np.random.uniform()
                    if(r < p):
                        child.con[eff][j] = donor.con[eff][j]
                    else:
                        child.con[eff][j] = original.con[eff][j]
            # Categorical
            if(np.random.uniform() < p):
                child.cat[eff] = donor.cat[eff]
            else:
                child.cat[eff] = original.cat[eff]
            # Mix
            if(np.random.uniform() < p):
                child.mix[eff] = donor.mix[eff]
            else:
                child.mix[eff] = original.mix[eff]

        return child
    
    def fitness_assessment(self,pop, idx=False):
        """Assess the fitness of a population, based on the eculidean norm of the difference between the 
            target spectrogram and the spectrogram obtained by an individual.
        
        ---
        Parameters
        - pop: population to evaluate
        - idx: Choose wether to return just the fitness or a list of lists containing the index of the individual and its fitness value"""
        n = len(pop)
        fitness = np.zeros(n)
        if idx:
            fitness = []
            for i in range(n):
                utils.set_parameters(pop[i].parameters,base_board)
                base_board.reset()
                effected = base_board(self.input,self.sr)
                effected = base_board(self.input,self.sr)
                utils.clip_signal(effected)
                S_eff = spectrogram.from_array(effected,self.hop_len)
                fitness.append([np.linalg.norm(self.target-S_eff), i])
        else: 
            for i in range(n):
                utils.set_parameters(pop[i].parameters,base_board)
                base_board.reset()
                effected = base_board(self.input,self.sr)
                effected = base_board(self.input,self.sr)
                utils.clip_signal(effected)
                S_eff = spectrogram.from_array(effected,self.hop_len)
                fitness[i] = np.linalg.norm(self.target-S_eff)
                
        return fitness
    def HJ_opt(self,ind,fitness_parent,fitness_children,alpha,beta, n_attempts=3, lamarckian=True):
        """Implements a modified version of Hooke-Jeeves optimization, 
            doing random jumps for each parameter between (-alpha,alpha).
        
        ---
        Parameters
        
        - ind: individual to optimize
        - fitness_parent: fitness value of the parent
        - fitness_children: fitness of the individual before the optimization
        - alpha: determines the maximum width of the jump to make in the optimization. Note that the optimization takes place using 
        the standardized values (thus between 0 and 1).
        - beta: regulates the strength of the pattern move of the optimization
        - n_attempts: number of iterations of the algorithm, at each iteration if the fitness is still higher than the 
        parent's one it will divide alpha by a factor of 2.
        - lamarckian: wether to apply lamarckian learning (return the optimized individual and its fitness) or baldwinian learning 
        (return the original individual but with the fitness of the optimized one)"""
        flag = True
        attempts = 0
        ind_search = ind.con.copy()
        while flag:
            for i in range(3):
                if ind.mix[i] > 0:
                    lst = par_con[i]
                    names = list(base_board[i].parameters.keys())
                    for j in range(len(ind_search[i])):
                        parameter = base_board[i].parameters[names[lst[j]]]
                        top = parameter.range[1]
                        bottom = parameter.range[0]
                        curr_val = utils.minmax_std(ind.con[i][j],bottom,top)
                        new_val = utils.rev_minmax(curr_val + np.random.uniform(-alpha,alpha),bottom,top)
                        ind_search[i][j] = new_val
            self.clip_parameters(ind_search,base_board)
            utils.set_parameters([ind_search,ind.cat,ind.mix],base_board)
            base_board.reset()
            effected = base_board(self.input,self.sr)
            effected = base_board(self.input,self.sr)
            utils.clip_signal(effected)
            S_eff = spectrogram.from_array(effected,self.hop_len)
            fitness_exp = np.linalg.norm(self.target-S_eff)
            if(fitness_exp<fitness_parent):
                flag = False
            elif attempts > n_attempts:
                return fitness_children
            else:
                attempts += 1
                alpha /= 2
        flag = True
        attempts = 0
        ind_patt = ind_search.copy()
        
        for i in range(3):
            ind_patt[i] += beta*(ind_search[i]-ind.con[i])
        self.clip_parameters(ind_search,base_board)
        utils.set_parameters([ind_search,ind.cat,ind.mix],base_board)
        base_board.reset()
        effected = base_board(self.input,self.sr)
        effected = base_board(self.input,self.sr)
        utils.clip_signal(effected)
        S_eff = spectrogram.from_array(effected,self.hop_len)
        fitness_patt = np.linalg.norm(self.target-S_eff)
        if(fitness_patt<fitness_parent):
            if(lamarckian):
                ind.con = ind_patt
            return fitness_patt
        else:
            if lamarckian:
                ind.con = ind_search
            return fitness_exp
           

class DE(Base):
    def rand1mutation(self,pop,idx,alpha):
        donors_idx = np.random.choice(np.delete(np.arange(len(pop)),idx),3)
        a = pop[donors_idx[0]]
        b = pop[donors_idx[1]]
        c = pop[donors_idx[2]]
        # Continuous parameters
        d_con = [0,0,0]
        d_mix = [0,0,0]
        for i in range(3):
            d_con[i] = a.con[i] + alpha*(b.con[i] - c.con[i])
            d_mix[i] = a.mix[i] + alpha*(b.mix[i] - c.mix[i])
        # Categorical parameters
        donors = [a,b,c]
        d_cat = donors[np.random.randint(0,3)].cat
        d_par = [d_con,d_cat,d_mix]  
        return d_par
    
    def generation_rand1(self,pop,alpha_mutation,p_crossover):
        children = []
        
        for i in range(len(pop)):
            donor = individual()
            donor.parameters = self.rand1mutation(pop,i,alpha_mutation)
            child = self.unif_crossover(pop[i],donor,p_crossover)
            self.clip_parameters(child.con, base_board)
            children.append(child)
        return children
    
    def evolve(self,n_gen,pop,alpha_mutation,p_crossover, alpha,decay,beta,n_try, verboso = True):
        """Evolve a population using DE/rand/1 with HJ local optimization.

        ---
        Parameters

        - n_gen: number of generations until algorithm stops
        - pop: population to evolve
        - alpha_mutation: mutation factor for the donor vector
        - p_crossover: probability of swapping genomes during the uniform crossover
        - alpha: determines the maximum width of the jump to make in the optimization phase. Note that the optimization takes place using 
        the standardized values (thus between 0 and 1).
        - decay: additional decay to decrease the alpha value as generation increase for fine optimization in the last generations
        - beta: regulates the strength of the pattern move of the optimization phase
        - n_try: number of iterations of the optimization algorithm, at each iteration if the fitness is still higher than the 
        parent's one it will divide alpha by a factor of 2
        - verboso: prints additional information of the evolution process every 5 generations
        """
        n = len(pop)
        fitness_curr = np.zeros(n)
        fitness_next = np.zeros(n)

        fitness_curr = DE.fitness_assessment(self,pop)
        best_idx = fitness_curr.argmin()
        for epoch in range(n_gen):
            children = self.generation_rand1(pop,alpha_mutation,p_crossover)
            fitness_next = self.fitness_assessment(children)
            alpha_hj = alpha * (decay)**epoch 
            beta_topass = beta
            for ind in range(n):
                if fitness_curr[ind]< fitness_next[ind]:
                    if(np.random.uniform()<(epoch/n_gen)):
                        fitness_next[ind] = self.HJ_opt(children[ind],fitness_curr[ind],fitness_next[ind],alpha_hj,beta_topass,n_try)
                else: 
                    pop[ind].parameters = children[ind].parameters
                    fitness_curr[ind] = fitness_next[ind]
                

            if((epoch)%5 == 0) and verboso:
                print("Gen.", epoch, "| Best fit:", fitness_curr.min(), "| Avg. Fitness: ", fitness_curr.mean(), "| Std. Fitness: ", fitness_curr.std())
        best_idx = fitness_curr.argmin()
        if verboso:
            print("\nBest overall individual:", best_idx, ", Fitness:", fitness_curr[best_idx])
        return pop[best_idx]   

class JADE_arc(Base):
    """Implements the JADE algorithm with archive."""
    
    def jademutation(self,pop,idx,best,archive,alpha_mutation):
        best_donor = np.random.choice(best)
        donor_b = np.random.choice(np.delete(np.arange(len(pop)), idx))
        donor_c = np.random.choice(np.delete(np.arange(len(pop)+len(archive)), idx))
        a = pop[best_donor]
        b = pop[donor_b]
        c = 0
        if donor_c >= len(pop):
            c = archive[donor_c-len(pop)]
        else:
            c = pop[donor_c]

        # Continuous parameters
        d_con = [0, 0, 0]
        d_mix = [0, 0, 0]
        for i in range(3):
            d_con[i] = pop[idx].con[i] + alpha_mutation*(a.con[i] - pop[idx].con[i]) + alpha_mutation*(b.con[i] - c.con[i])
            d_mix[i] = pop[idx].mix[i] + alpha_mutation*(a.mix[i] - pop[idx].mix[i]) + alpha_mutation*(b.mix[i] - c.mix[i])
        # Categorical parameters
        donors = [a,b,c]
        d_cat = donors[np.random.randint(0,3)].cat 
    
        d_par = [d_con, d_cat, d_mix]
        return d_par
    def generation(self,pop,fitness,archive,alpha_array,p_array, p_best):
        children = []
        for i in range(len(pop)):
            best =  [x[1] for x in fitness[:p_best] if x[1] != i] 
            donor = individual()
            donor.parameters = self.jademutation(pop,i,best,archive,alpha_array[i])
            child = self.unif_crossover(pop[i],donor,p_array[i])
            self.clip_parameters(child.con, base_board)
            children.append(child)
        return children
    
    def evolve(self,n_gen,pop,alpha_mutation,p_crossover, p_best,c, dim_arc, alpha,decay,beta,n_try,p_lamarckian,verboso = False):
        """Evolve a population using the JADE algorithm with HJ local optimization.

        ---
        Parameters

        - n_gen: number of generations until algorithm stops
        - pop: population to evolve
        - alpha_mutation: starting mutation factor for the donor vector
        - p_crossover: starting probability of swapping genomes during the uniform crossover
        - p_best: number of p top-performing individuals to use in the crossover process
        - c: "learning rate" for the mean of p_crossover and alpha_mutation distributions 
        - dim_arc: dimension of the archive
        - alpha: determines the maximum width of the jump to make in the optimization phase. Note that the optimization takes place using 
        the standardized values (thus between 0 and 1).
        - decay: additional decay to decrease the alpha value as generation increase for fine optimization in the last generations, should be in [0,1]
        - beta: regulates the strength of the pattern move of the optimization phase
        - n_try: number of iterations of the optimization algorithm, at each iteration if the fitness is still higher than the 
        parent's one it will divide alpha by a factor of 2
        - verboso: prints additional information of the evolution process every 5 generations
        """
        n = len(pop)
        fitness_curr = []
        fitness_next = np.zeros(n)
        archive = []
        fitness_curr = self.fitness_assessment(pop,True)
        fitness_curr.sort()
        mu_p = 0.
        mu_alpha_num = 0.
        mu_alpha_den = 0.
        p_array = np.zeros(n)
        alpha_array = np.zeros(n)
        
        for generation in range(n_gen):
            i = 0
            while i < n:
                tmp = np.random.normal(p_crossover,.31623) # sqrt(.1)
                if (tmp < 0) or (tmp > 1):
                    pass
                else:
                    p_array[i] = tmp
                    i+=1
            i = 0
            while i < n:
                tmp =  np.random.standard_cauchy()*.1 + alpha_mutation
                if (tmp < 0):
                    pass
                elif (tmp > 1): 
                    alpha_array[i] = 1
                    i+=1
                else:
                    alpha_array[i] = tmp
                    i+=1

            children = self.generation(pop,fitness_curr,archive,alpha_array,p_array, p_best)
            fitness_next = self.fitness_assessment(children)
            fitness_curr = sorted(fitness_curr, key=lambda x: x[1])
            alpha_hj = alpha * (decay)**generation
            beta_topass = beta
            for ind in range(n):
                if fitness_curr[ind][0] < fitness_next[ind]:
                    if(np.random.uniform()<(generation/n_gen)):
                        if(np.random.uniform() < p_lamarckian):
                            fitness_next[ind] = self.HJ_opt(children[ind],fitness_curr[ind][0],fitness_next[ind],alpha_hj,beta_topass,n_try)
                        else:
                            fitness_next[ind] = self.HJ_opt(children[ind],fitness_curr[ind][0],fitness_next[ind],alpha_hj,beta_topass,n_try,False)
                if fitness_curr[ind][0] > fitness_next[ind]:
                    if(len(archive) < dim_arc + 1):
                        archive.append(pop[ind])
                    else:
                        index = np.random.randint(0,dim_arc)
                        archive[index] = pop[ind]
                    pop[ind].parameters = children[ind].parameters
                    fitness_curr[ind][0] = fitness_next[ind]
                    mu_p += p_array[ind]
                    mu_alpha_num += alpha_array[ind]**2
                    mu_alpha_den += alpha_array[ind]
            if(mu_p > 0.):
                p_crossover = (1-c)*p_crossover + c*(mu_p/n)
                alpha_mutation = (1-c)*alpha_mutation + c*(mu_alpha_num/mu_alpha_den)
                mu_p = 0.
                mu_alpha_num = 0.
                mu_alpha_den = 0.

            fitness_curr.sort()
            best_idx = fitness_curr[0][0]
            if((generation)%5 == 0) and verboso == True:
                mu = 0.
                std = 0.
                for x in range(n):
                    mu += fitness_curr[x][0]
                mu /=n
                for x in range(n):
                    std += (fitness_curr[x][0]-mu)**2
                std = np.sqrt(std/(n-1))
                print("Gen.", generation, "| Best fit:", fitness_curr[0][0], "| Avg. Fitness:", 
                        mu, "| Std. Fitness: ", std, "| alpha_mutation:", '{:4.3f}'.format(alpha_mutation), "| p_crossover:",'{:4.3f}'.format(p_crossover) )
        
        best_idx = fitness_curr[0][1]
        if(verboso == True):
            print("\nBest overall individual:", best_idx, ", Fitness:", fitness_curr[0][0])
        return pop[best_idx]      