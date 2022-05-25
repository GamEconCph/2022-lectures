# bimatrix: a collection of functions relevant for working 
# with bimatrix games. 
import pandas as pd 
import numpy as np
import copy
import matplotlib.pyplot as plt
from scipy.optimize import linprog 

def solve_zerosum_with_linprog(U):
    '''solve_zerosum_with_linprog(): Solve a zero sum game using linear programming
    
        INPUT: U (k*k square matrix), payoffs in zero sum game (opponent gets -U.T)
        OUTPUT: alpha (k-vector) of probability weights for each action (the symmetric equilibrium)
    '''
    k, k2 = U.shape
    assert k == k2, f'Input matrix must be square, got {k}*{k2}'

    oo = np.zeros((1,k))
    ii = np.ones((1,k))

    # objective: c = [-1, 0, 0, ..., 0]
    c = np.insert(oo, 0, -1.0) # insert -1 in front (pos = index 0)
    
    # inequality constraints A*x <= b
    # top = [ 1 ...
    #         1 -1*A.Tl
    #         1  ...  ]
    # bot = [ 0 -1 0 0 
    #         0 0 -1 0 
    #         0 0 0 -1]
    top  = np.hstack( (ii.T, -1*U.T) )
    bot  = np.hstack( (oo.T, -1*np.eye(k)) )
    A_ub = np.vstack((top, bot))
    
    b_ub = np.zeros((1, 2*k))
    b_ub = np.matrix(b_ub)
    
    # contraints Ax = b
    # A = [0, 1, 1, ..., 1]
    A = np.matrix(np.hstack((0, np.ones((k,)))))
    b = 1.0 # just one condition so scalar 

    # v and alpha must be non-negative
    bounds = [(0,None) for i in range(k+1)]

    # call the solver
    sol = linprog(c=c, A_ub=A_ub, b_ub=b_ub, A_eq=A, b_eq=b)
    
    # remove the first element: just return the Nash EQ 
    alpha = sol.x[1:]
    return alpha


def best_response(U, i): 
    """best_response(): 
        INPUTS: 
            U: list of payoff matrices 
            i: (int) player for whom to do the best response 

        OUTPUT: 
            br: (NEQ*2) matrix, where br[:,0] is opponent strategies
                and br[:,1] are the best responses. If one strategy a
                has multiple best responses, then there will be several
                columns in br with br[:,0]==a. 
    """
    j = 1-i # opponent
    if i == 0: 
        Ui = U[0]
    elif i == 1: 
        Ui = U[1].T # so that i becomes row player 
    else: 
        raise Exception(f'Not implemented for n>2 players, got i={i}')

    nai, naj = Ui.shape

    # initialie 
    br = []

    for aj in range(naj):
        # column of U corresponding to the aj'th action of the opponent
        Ui_j = Ui[:, aj] 

        # find index values for the rows where Ui_j attains the max
        idim = 0 # there will not be more dimensions in our case 
        br_ij = np.where(Ui_j == Ui_j.max())[idim]

        for b in br_ij: 
            br.append([aj, b])

    return np.array(br)
    

def print_payoffs(U, A, round_decimals=None): 
    '''print_payoffs: Nicely formatted for a 2*2 game 
        INPUTS: 
            U1,U2: (matrices, dim=na1*na2) Payoffs 
            A1: (list of str, len=na1) List of actions of player 1
            A2: (list of str, len=na2) list of actions of player 2
            round_decimals: (int) Number of decimals of precision to print with 
        
        OUTPUT:
            tab: pandas dataframe, na1*na2 with payoff tuples 
    '''
    assert len(U) == 2, f'only implemented for 2-player games'
    assert len(A) == 2, f'only implemented for 2-player games'

    U1 = U[0]
    U2 = U[1]
    A1 = A[0]
    A2 = A[1]

    na1,na2 = U1.shape
    assert len(A1) == na1
    assert len(A2) == na2

    if not (round_decimals is None):
        assert np.isscalar(round_decimals), f'round_decimals must be an integer' 
        U1 = U1.round(round_decimals)
        U2 = U2.round(round_decimals)

    # "matrix" of tuples 
    X = [[(U1[r,c],U2[r,c]) for c in range(na2)] for r in range(na1)]

    # dataframe version 
    tab = pd.DataFrame(X, columns=A2, index=A1)
    
    return tab 

def find_undominated_actions(U_in, i, A, DOPRINT=False):
    '''find_undominated_actions: finds the actions for player i that are
        not strictly dominated by another action
        
        INPUTS: 
            U_in: (matrix, na1*na2) Payoffs (player 1, player 2)
            i: (integer) Which player we are currently examining
            A: (list) List of actions (len = # of actions for this player)
            
        OUTPUT: 
            AA: (list) undominated actions 
            IA: (list of integers) integers i s.t. AA = [A[i] for i in IA]
            ANYDOMINATED: (bool) True if at least one action was strictly dominated
    '''
    
    AA = []
    IA = []
    nA = len(A)
    
    # 1. ensure that U has actions of player i along 0th dimension 
    if i == 0: 
        # 1.a already the case 
        U = np.copy(U_in)
    else: 
        # 1.b transpose 
        U = U_in.T 
    
    # 2. determine if each action has other dominated actions 
    for ia in range(nA): 
        DOMINATED = False 
                
        for ia_ in range(nA): 
            # 2.a loop through all *other* strategies 
            if ia_ == ia: 
                continue

            # 2.b check if ia_ always gives a higher payoff than ia (i.e. domination)
            if np.all(U[ia_] > U[ia]): 
                DOMINATED = True
                break # exit search: enough that we have found one 
        
        # 2.c append or not 
        if not DOMINATED: 
            AA.append(A[ia])
            IA.append(ia)
            
    # 3. convenient boolean 
    ANYDOMINATED = (len(AA) < len(A))
    
    return AA,IA,ANYDOMINATED


def IESDS(A, U, DOPRINT=False, maxit=10000): 
    '''Iterated Elimination of Strictly Dominated Strategies 
        INPUTS: 
            A: (list of lists) n lists (one for each player), 
                    each has len = # of actions to player i
            U: (list, len=n) list of na1*na2 matrices of payoffs
            DOPRINT: (bool) whether to print output to terminal 
            maxit: (int) break algorithm if this count is ever reached
                (note: the algorithm is not approximate so we can compute 
                what maxit is in the worst case)
        OUTPUT: Actions and payoffs for the undominated game
            A_undominated: (n-list of vectors) 
            U_undominated: (n-list of matrices of payoffs)
    '''
    
    U_undominated = copy.copy(U)
    A_undominated = copy.copy(A)
    
    n = len(U)
    na1,na2 = U[0].shape

    # checks 
    assert n == 2, f'Code only implemented for 2-player games '
    assert len(A) == n
    for i in range(n): 
        assert len(A[i]) == U[i].shape[i]
        assert U[i].shape == (na1,na2), f'Payoff matrix for player {i+1} is {U[i].shape}, but {(na1,na2)} for player 1'

    # initialize flags 
    D = np.ones((n,), dtype='bool')
    
    for it in range(maxit): 

        for i in range(n): 
            # find undominated actions 
            A_undominated[i], IA, D[i] = find_undominated_actions(U_undominated[i], i, A_undominated[i], DOPRINT)

            # if we found at least one, remove it/them from the game 
            if D[i]: 
                # remove from both players' payoff matrices 
                for j in range(n): 
                    if i == 0: 
                        U_undominated[j] = U_undominated[j][IA, :]
                    else: 
                        U_undominated[j] = U_undominated[j][:, IA]


        # break once we have run an iteration without finding any strategies to remove 
        if D.any() == False: 
            break

    return A_undominated, U_undominated
