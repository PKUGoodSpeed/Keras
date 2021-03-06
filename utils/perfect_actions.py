'''
Here we implement function or classes to compute the differently defined perfect action,
for a changing first level ask price and bid price.
The first one is a safe strategy, which will always gets profit, but not guarantee the total profit.
'''

import numpy as np
from progress import ProgressBar

class PerfectAction:
    ''' 
    For computing differently defined perfect actions:
    All actions are classified as:
        0: holding (doing nothing)
        1: buying
        2: selling
    
    1: Safe action: but it gives a very bad pnl even though it is positive
    '''
    __ask = None
    __bid = None
    
    def __init__(self, ask_price, bid_price):
        '''Getting price information'''
        self.__ask = np.array(ask_price)
        self.__bid = np.array(bid_price)
        assert len(self.__ask) == len(self.__bid)
    
    def getPnl(self, actions = None, fee = 0.41):
        '''
        Given a particular action rule, get the pnl from it.
        '''
        if actions is None:
            return 0.
        assert len(actions) == len(self.__ask);
        position = 0
        total_pnl = 0.
        last_price = 0.
        for i in range(len(self.__ask)):
            if actions[i] == 1 and position < 1:
                total_pnl -= (1-position) * self.__ask[i] + fee
                position = 1
                last_price = self.__ask[i]
            elif actions[i] == 2 and position >-1:
                total_pnl += (1+position) * self.__bid[i] - fee
                position = -1
                last_price = self.__bid[i]
        total_pnl += position * last_price
        return total_pnl
        
    def naiveSafeAction(self, tick_size = 1.0, prof = 5.):
        '''
        The action defined in this way guarantees that every buy/sell can gets profit >= prof
        If we include the fee, we need to 
        '''
        low = np.min(self.__bid)
        high = np.max(self.__ask)
        while high > low + tick_size:
            mid = 0.5 * (high + low)
            num_buy = (self.__ask < mid).sum()
            num_sell = (self.__bid > mid + prof).sum()
            if num_buy > num_sell:
                high = mid
            else:
                low = mid
        mid = 0.5 * (high + low)
        return np.int16(self.__ask < mid) + np.int16(self.__bid > mid + prof)*2
        
    def bestSafeAction(self, tick_size = 1.0, prof_range = (1., 100.), fee = 0.41):
        '''
        Choose the Safe Action label which can return as the highest pnl
        '''
        pb = ProgressBar()
        best_pnl = 0.
        best_prof = 0.
        profs = np.arange(prof_range[0], prof_range[1], 0.5*tick_size)
        pb.setBar(num_iteration = len(profs), bar_size = 100)
        for i in range(len(profs)):
            pb.show(i)
            prof = profs[i]
            actions = self.naiveSafeAction(tick_size = tick_size, prof = prof)
            tmp_pnl = self.getPnl(actions, fee = fee)
            if tmp_pnl > best_pnl:
                best_pnl = tmp_pnl
                best_prof = prof
        return best_pnl, best_prof, self.naiveSafeAction(tick_size = tick_size, prof = best_prof)