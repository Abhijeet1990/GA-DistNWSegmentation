
"""
Test bed for experimenting with simulated annealing (SA).

SA is a simple heuristic search algorithm that works quite well in practice
on many theoretically intractable problems such as travelling salesman (TSP).
For more about SA, see The Algorithm Design Handbook by Skiena.
To sum it up in Skiena's words:
"In expert hands, the best problem-specific heuristics for TSP can slightly
outperform simulated annealing. But the simulated annealing solution
works admirably. It is my heuristic method of choice."

SA is very simple to describe, but actual implementations may involve
a few subtle points not included in a typical description, so this code
may be useful to those wanting to try SA out on a particular problem.
In outline, SA works as follows:
    - start with a random guess at a solution, and a virtual temperature
        that we lower gradually as the search continues
    - randomly mutate the guess
    - if the new guess is an improvement, accept it and continue
    - otherwise (the new guess is worse than the current one), accept the new
        guess with probability based on the current temperature (the lower
        it is, the less likely we accept), and on how much worse the new
        guess is (the worse it is, the less likely we accept)
    - continue until the search appears to be frozen

The code below is my attempt to reverse-engineer the experiments in:
Optimization by Simulated Annealing: An Experimental Evaluation;
Part I, GraphPartitioning,
by Johnson et.al.,
Operations Research, Vol. 37, No. 6 (Nov. - Dec., 1989), pp. 865-892
http://www.jstor.org/stable/171470

As an example, SA is applied below to the graph partition problem for
randomly generated large graphs. In this problem, we're given a graph with
an even number of vertices, and want to split the vertices into two halves
such that we minimize the number of edges with vertices in opposite halves.
This is an NP-complete problem, and so theoretically intractable.
"""

import itertools
import math
import random
import sys

def main() :
    """
    Run a simple test - a simulated annealing search on a particular generated
    graph partition problem.
    """
    problem123 = graphPartitionProblem.makeRandomProblem(n=500,p=0.01,rng=123)
    saSearch(problem123,4,rng=323)


# example Problem object
# This defines the methods needed to call the general SA code below.
class graphPartitionProblem(object) :
    """
    Test class representing an undirected graph partition problem.
    Graphs have n vertices 0..n-1, and are represented by a list of
    edges [(v1,v2), (v3,v4), ... ].

    Note that although our final answer is required to be balanced (both sides
    of the partition must have the same number of vertices), we will allow
    guessed partitions to become unbalanced during the search. This is common
    in SA where the final answer must fulfill some constraints - we might want
    to relax the constraints during search to make it more efficient or simpler
    to program. This is handled below by applying a cost penalty to unbalanced
    guesses, and using a simple greedy heuristic to rebalance final guesses.
    """
    def __init__(self, n, edges) :
        """
        Creates a problem instance with specified edges.
        """
        self.n = n
        self.edges = edges
    @classmethod
    def makeRandomProblem(cls, n=500, p=0.01, rng=None) :
        """
        Generates an undirected random graph with n vertices 0..n-1,
        and probability p that each edge exists (decided independently).
        """
        rng = getRng(rng)
        edges = [edge for edge in itertools.combinations(range(n),2)
                        if rng.random() <= p]
        return cls(n,edges)
    def getCost(self,s,alpha) :
        """
        Calculates cost of a guessed partition represented as a set of vertices.
        Alpha is a constant used to impose a penalty on unbalanced partitions.
        """
        return (sum(1 for u,v in self.edges if (u in s) != (v in s))
                + alpha*(len(s)-(self.n-len(s)))**2)
    def getRandomGuesses(self,nGuesses,rng) :
        """
        Returns a list of n unique guessed partitions represented as
        frozensets of vertices.
        """
        vList = range(self.n)
        res = set()
        while len(res) < nGuesses :
            res.add(frozenset(rng.sample(vList,int(self.n/2))))
        return list(res)
    def mutateGuess(self,s,rng) :
        """
        Randomly changes a guessed partition represented as a set of vertices.
        """
        return s.symmetric_difference((rng.randint(0,self.n-1),))
    def guessSatisfiesConstraints(self,s) :
        """
        Says if a guessed partition is balanced as required in our final answer.
        """
        return (len(s) == self.n/2)
    def modifyGuessToSatisfyConstraints(self,s,alpha) :
        """
        Greedy heuristic to rebalance a possibly unbalanced guessed partition
        represented as a set of vertices. Starts with the larger half and keeps
        dropping the vertex that gives the lowest cost result.
        """
        if len(s) < self.n/2 :
            res = set(range(self.n)) - s
        else :
            res = set(s)
        while len(res) > self.n/2 :
            bestCost = bestV = None
            for v in set(res) :
                res.remove(v)
                curCost = self.getCost(res,alpha)
                if bestCost is None or curCost<bestCost :
                    bestCost = curCost
                    bestV = v
                res.add(v)
            res.remove(bestV)
        return frozenset(res)
    def getTempLength(self) :
        """
        Returns the number of guesses made at each temperature in the SA search
        if not specified in the search call.
        """
        return self.n*16


# General simulated annealing code:
def getRng(rng) :
    """
    Used to get a random number generator to be used below.
    We might want to use a specific seed for repeatability.
    If rng is an int or long, returns a new generator seeded with that value.
    If rng is None, returns a new randomly seeded generator.
    Otherwise assumes rng is a random number generator and returns it unchanged.
    """
    if rng is None or isinstance(rng,int):
        rng = random.Random(rng)
    return rng
def saAccept(curCost,nextCost,T,rng) :
    """
    Says whether to accept a transition from a guess with cost curCost to
    a mutated guess with cost nextCost, where T is the virtual temperature.
    This uses the traditional SA equation based on the Boltzmann distribution
    of energy differences at a given temperature.
    """
    return (curCost>=nextCost    # always accept better (lower-cost) guesses
            or rng.random()<=math.exp(-(nextCost-curCost)/T))
            # accept worse solutions with probability dependent on the
            # virtual temperature T and how much worse the mutated guess is
def saCalcStartTemp(problem, alpha=0.05,
                    initAcceptProb=0.4, nTestGuesses=500, rng=None) :
    """
    Automatically calculates a starting virtual temperature so that in an
    initial set of randomly guessed solutions, the probability of accepting
    a transition to a worse guess is initAcceptProb. This can calculate
    a reasonable starting temperature for a particular cost function.
    """
    # get a set of random guesses, mutate them, and get the costs:
    rng = getRng(rng)
    testCostsAndNextCosts = []
    for guess in problem.getRandomGuesses(nTestGuesses,rng) :
        nextGuess = problem.mutateGuess(guess,rng)
        cost = problem.getCost(guess,alpha)
        nextCost = problem.getCost(nextGuess,alpha)
        if nextCost > cost :
            testCostsAndNextCosts.append((cost,nextCost))
    #print testCostsAndNextCosts
    # do a binary search for the right virtual temperature:
    tempL = 0.0
    tempU = 10.0
    while (tempU-tempL > 0.01) :
        tempM = (tempU+tempL)/2.0
        nAccepted = sum(1 for cost,nextCost in testCostsAndNextCosts
                        if saAccept(cost,nextCost,tempM,rng))
        if nAccepted < initAcceptProb*len(testCostsAndNextCosts) :
            # too few accepted - need higher temp
            tempL = tempM
        else :
            # too many accepted - need lower temp
            tempU = tempM
    return tempU
def saSearch(problem, nRuns, tempLength=None,
             startTemp=None, initAcceptProb=0.4, nTestGuesses=500,
             tempFactor=0.95, alpha=0.05, minAcceptFrac=0.02,
             rng=None) :
    """
    Do a simulated annealing search on a problem.
    Returns a list [(bestCost, bestGuess), ... ] of the best solution found
    on each run of the search.

    Arguments:
    nRuns - number of search runs
    tempLength - number of guesses at each virtual temperature
        if None, calls problem.getTempLength() for this value
    startTemp, initAcceptProb, nTestGuesses - specify the initial temperature
        if startTemp is not None, this value is used
        if startTemp is None, saCalcStartTemp is called to calculate it
    tempFactor - factor the virtual temperature is multiplied by to get
        the next temperature value in the search loop
    alpha - penalty constant used to calculate cost for guesses that don't
        satisfy the problem constraints
    minAcceptFrac - search is decided to be frozen and ended when the fraction
        of accepted guesses is below minAcceptFrac for 5 straight temperatures,
        EXCEPT we reset the counter if a new best solution is found
    rng - specifies the random number generator to be obtained using getRng
        used for test repeatability
    """
    rng = getRng(rng)
    if startTemp is None :
        startTemp = saCalcStartTemp(problem,alpha,
                                    initAcceptProb,nTestGuesses,rng)
        print ('initAcceptProb '+str(initAcceptProb)+' startTemp '+str(startTemp))
    if tempLength is None :
        tempLength = problem.getTempLength()
    res = []
    for i in range(nRuns) :
        bestGuess = guess = problem.getRandomGuesses(1,rng)[0]
        bestCost = curCost = problem.getCost(guess,alpha)
        frozenCounter = 0  # used to check if the search appears frozen
        T = startTemp
        tempNo = 0
        while frozenCounter < 5 :
            print('T#{}: {:.3f} best {}'.format(tempNo,T,bestCost))
            tempNo += 1
            nAccepted = 0
            for j in range(tempLength) :
                nextGuess = problem.mutateGuess(guess,rng)
                nextCost = problem.getCost(nextGuess,alpha)
                if saAccept(curCost,nextCost,T,rng) :
                    # accept the move
                    guess,curCost = nextGuess,nextCost
                    nAccepted += 1
                    if (bestCost>curCost
                            and problem.guessSatisfiesConstraints(guess)) :
                        bestGuess,bestCost = guess,curCost
                        frozenCounter = 0
            if nAccepted <= minAcceptFrac*tempLength :
                frozenCounter += 1
            T *= tempFactor
        if bestCost > curCost :
            guess = problem.modifyGuessToSatisfyConstraints(guess,alpha)
            curCost = problem.getCost(guess,alpha)
            if bestCost > curCost :
                bestGuess,bestCost = guess,curCost
        res.append((bestCost,bestGuess))
        print('iter {}/{}: {}'.format(i+1,nRuns,res[-1][0]))
    return res

if __name__ == '__main__' :
    sys.exit(main())